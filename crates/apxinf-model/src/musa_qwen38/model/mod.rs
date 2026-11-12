//! Eager, one-token MUSA Qwen3.8 GDN composition.

pub mod mlp;
pub mod attention;

use apxinf_core::{DType, Error, Result, Shape, Tensor};
use apxinf_musa::{kernels::qwen38, ops, MusaContext};
use half::bf16;

const HIDDEN: usize = 5120;
const QKV: usize = 10240;
const Z: usize = 6144;
const AB: usize = 48;
const V_HEADS: usize = 48;
const K_HEADS: usize = 16;
const HEAD_DIM: usize = 128;
const CONV_WIDTH: usize = 4;
const EPSILON: f32 = 1e-6;

/// Device-resident BF16 weights for one GDN layer. Linear weights use GEMM
/// layout `[in, out]`; checkpoint mapping is intentionally not part of this
/// eager block API.
pub struct Qwen38GdnBf16Weights {
    /// Raw HF offset weight: the block applies (1 + input_norm).
    pub input_norm: Tensor,
    pub qkv: Tensor,
    pub z: Tensor,
    pub a: Tensor,
    pub b: Tensor,
    pub conv: Tensor,
    pub a_log: Tensor,
    pub dt_bias: Tensor,
    pub gdn_norm: Tensor,
    pub out: Tensor,
}

impl Qwen38GdnBf16Weights {
    /// Consume raw official HF tensors for one GDN mixer and upload them.
    /// Other layers and the MLP are not part of this mixer API.
    pub fn from_hf_layer(
        ctx: &MusaContext,
        layer: usize,
        mut tensors: std::collections::HashMap<String, Tensor>,
    ) -> Result<Self> {
        if layer >= 64 || (layer + 1) % 4 == 0 {
            return Err(Error::Other(format!("layer {layer} is not a supported GDN layer")));
        }
        let mut take = |suffix: &str, shape: &[usize], transpose: bool| -> Result<Tensor> {
            let key = format!("model.language_model.layers.{layer}.{suffix}");
            let tensor = tensors.remove(&key).ok_or_else(|| Error::Other(format!("missing {key}")))?;
            if tensor.dtype() != DType::BF16 || tensor.shape().dims() != shape {
                return Err(Error::Other(format!("{key}: expected BF16 {shape:?}, got {:?} {:?}", tensor.dtype(), tensor.shape().dims())));
            }
            let tensor = if transpose {
                let source = tensor.as_bf16()?;
                let (rows, cols) = (shape[0], shape[1]);
                let mut result = vec![bf16::ZERO; source.len()];
                for r in 0..rows { for c in 0..cols { result[c * rows + r] = source[r * cols + c]; } }
                Tensor::from_bf16(vec![cols, rows], &result)?
            } else { tensor };
            apxinf_musa::transfers::to_device(&tensor, ctx.device_id())
        };
        Ok(Self {
            input_norm: take("input_layernorm.weight", &[HIDDEN], false)?,
            qkv: take("linear_attn.in_proj_qkv.weight", &[QKV, HIDDEN], true)?,
            z: take("linear_attn.in_proj_z.weight", &[Z, HIDDEN], true)?,
            a: take("linear_attn.in_proj_a.weight", &[AB, HIDDEN], true)?,
            b: take("linear_attn.in_proj_b.weight", &[AB, HIDDEN], true)?,
            conv: take("linear_attn.conv1d.weight", &[QKV, 1, CONV_WIDTH], false)?.reshape(vec![QKV, CONV_WIDTH])?,
            a_log: take("linear_attn.A_log", &[V_HEADS], false)?,
            dt_bias: take("linear_attn.dt_bias", &[V_HEADS], false)?,
            gdn_norm: take("linear_attn.norm.weight", &[HEAD_DIM], false)?,
            out: take("linear_attn.out_proj.weight", &[HIDDEN, Z], true)?,
        })
    }
}

/// Mutable F32 recurrent and convolution state for one GDN layer.
pub struct Qwen38GdnBf16State {
    pub recurrent: Tensor,
    pub conv_window: Tensor,
}

/// Eager first-layer GDN block. It allocates intermediates per call and does
/// not capture graphs or own a session. This is the GDN mixer plus residual,
/// not a complete transformer layer (no MLP). Sequence/prefill is unsupported; call
/// `forward_decode` once per token.
pub struct Qwen38GdnBf16Block {
    pub weights: Qwen38GdnBf16Weights,
}

fn zero_bf16(ctx: &MusaContext, dims: Vec<usize>) -> Result<Tensor> {
    let host = Tensor::from_bf16(
        dims.clone(),
        &vec![bf16::from_f32(0.0); dims.iter().product()],
    )?;
    apxinf_musa::transfers::to_device(&host, ctx.device_id())
}

fn zero_f32(ctx: &MusaContext, dims: Vec<usize>) -> Result<Tensor> {
    let host = Tensor::from_f32(dims.clone(), &vec![0.0; dims.iter().product()])?;
    apxinf_musa::transfers::to_device(&host, ctx.device_id())
}

impl Qwen38GdnBf16State {
    pub fn zeroed(ctx: &MusaContext) -> Result<Self> {
        Ok(Self {
            recurrent: zero_f32(ctx, vec![V_HEADS, HEAD_DIM, HEAD_DIM])?,
            conv_window: zero_f32(ctx, vec![QKV, CONV_WIDTH])?,
        })
    }
}

impl Qwen38GdnBf16Block {
    pub fn new(weights: Qwen38GdnBf16Weights) -> Self {
        Self { weights }
    }

    /// State is advanced in place; discard it after a device execution error.
    pub fn forward_decode(
        &self,
        ctx: &MusaContext,
        hidden: &Tensor,
        state: &mut Qwen38GdnBf16State,
    ) -> Result<Tensor> {
        if hidden.dtype() != DType::BF16 || hidden.shape().dims() != [1, HIDDEN] {
            return Err(Error::Other(format!(
                "Qwen3.8 GDN hidden must be BF16 [1, {HIDDEN}], got {:?} {:?}",
                hidden.dtype(),
                hidden.shape().dims()
            )));
        }
        for (name, tensor, shape, dtype) in [
            ("hidden", hidden, vec![1, HIDDEN], DType::BF16),
            (
                "input_norm",
                &self.weights.input_norm,
                vec![HIDDEN],
                DType::BF16,
            ),
            ("qkv", &self.weights.qkv, vec![HIDDEN, QKV], DType::BF16),
            ("z", &self.weights.z, vec![HIDDEN, Z], DType::BF16),
            ("a", &self.weights.a, vec![HIDDEN, AB], DType::BF16),
            ("b", &self.weights.b, vec![HIDDEN, AB], DType::BF16),
            (
                "conv",
                &self.weights.conv,
                vec![QKV, CONV_WIDTH],
                DType::BF16,
            ),
            ("a_log", &self.weights.a_log, vec![V_HEADS], DType::BF16),
            ("dt_bias", &self.weights.dt_bias, vec![V_HEADS], DType::BF16),
            (
                "gdn_norm",
                &self.weights.gdn_norm,
                vec![HEAD_DIM],
                DType::BF16,
            ),
            ("out", &self.weights.out, vec![Z, HIDDEN], DType::BF16),
            (
                "recurrent",
                &state.recurrent,
                vec![V_HEADS, HEAD_DIM, HEAD_DIM],
                DType::F32,
            ),
            (
                "conv_window",
                &state.conv_window,
                vec![QKV, CONV_WIDTH],
                DType::F32,
            ),
        ] {
            if tensor.shape().dims() != shape
                || tensor.dtype() != dtype
                || tensor.device() != apxinf_core::Device::Musa(ctx.device_id())
            {
                return Err(Error::Other(format!(
                    "Qwen3.8 GDN invalid {name}: expected {dtype:?} {shape:?} on context device"
                )));
            }
        }
        let mut normalized = ctx.allocate_output(Shape::new(vec![1, HIDDEN]), DType::BF16)?;
        qwen38::offset_rms_bf16(ctx, hidden, &self.weights.input_norm, &mut normalized, EPSILON)?;
        let qkv = gemm(ctx, &normalized, &self.weights.qkv, vec![1, QKV])?
            .reshape(Shape::new(vec![QKV]))?;
        let z = gemm(ctx, &normalized, &self.weights.z, vec![1, Z])?;
        let a = gemm(ctx, &normalized, &self.weights.a, vec![1, AB])?.reshape(vec![AB])?;
        let b = gemm(ctx, &normalized, &self.weights.b, vec![1, AB])?.reshape(vec![AB])?;
        let mut conv = ctx.allocate_output(Shape::new(vec![QKV]), DType::BF16)?;
        qwen38::gdn_causal_conv_step(
            ctx,
            &mut state.conv_window,
            &qkv,
            &self.weights.conv,
            &mut conv,
        )?;
        let mut q = view(&conv, 0, K_HEADS * HEAD_DIM, vec![K_HEADS, HEAD_DIM])?;
        let mut k = view(
            &conv,
            K_HEADS * HEAD_DIM,
            K_HEADS * HEAD_DIM,
            vec![K_HEADS, HEAD_DIM],
        )?;
        let v = view(
            &conv,
            2 * K_HEADS * HEAD_DIM,
            V_HEADS * HEAD_DIM,
            vec![V_HEADS, HEAD_DIM],
        )?;
        qwen38::gdn_l2_normalize_heads(ctx, &mut q, EPSILON)?;
        qwen38::gdn_l2_normalize_heads(ctx, &mut k, EPSILON)?;
        let mut decay = ctx.allocate_output(Shape::new(vec![V_HEADS]), DType::F32)?;
        let mut beta = ctx.allocate_output(Shape::new(vec![V_HEADS]), DType::F32)?;
        qwen38::gdn_decay_beta_step(
            ctx,
            &a,
            &b,
            &self.weights.a_log,
            &self.weights.dt_bias,
            &mut decay,
            &mut beta,
        )?;
        let mut readout = ctx.allocate_output(Shape::new(vec![V_HEADS, HEAD_DIM]), DType::BF16)?;
        qwen38::gdn_recurrent_step(
            ctx,
            &mut state.recurrent,
            &q,
            &k,
            &v,
            &decay,
            &beta,
            &mut readout,
        )?;
        let mut gated = ctx.allocate_output(Shape::new(vec![V_HEADS, HEAD_DIM]), DType::BF16)?;
        qwen38::gated_rms_bf16(ctx, &readout, &self.weights.gdn_norm,
            &z.reshape(vec![V_HEADS, HEAD_DIM])?, &mut gated, EPSILON)?;
        let mut output = gemm(
            ctx,
            &gated.reshape(Shape::new(vec![1, Z]))?,
            &self.weights.out,
            vec![1, HIDDEN],
        )?;
        qwen38::add_into(ctx, &mut output, hidden)?;
        Ok(output)
    }
}

fn gemm(ctx: &MusaContext, a: &Tensor, b: &Tensor, out_shape: Vec<usize>) -> Result<Tensor> {
    let mut out = ctx.allocate_output(Shape::new(out_shape), DType::BF16)?;
    ops::gemm(ctx, ops::GemmArgs::new(a, b, &mut out))?;
    Ok(out)
}

fn view(tensor: &Tensor, offset: usize, len: usize, shape: Vec<usize>) -> Result<Tensor> {
    apxinf_musa::MusaBuffer::from_tensor(tensor)
        .map_err(Error::Other)?
        .view(
            offset * DType::BF16.size_in_bytes(),
            len * DType::BF16.size_in_bytes(),
        )
        .map_err(Error::Other)?
        .as_tensor(Shape::new(shape), DType::BF16)
        .map_err(Error::Other)
}
