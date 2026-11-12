//! Eager BF16 MLP with HF offset RMSNorm and vendor SwiGLU.
use super::{gemm, HIDDEN, EPSILON};
use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_musa::{kernels, ops, transfers, MusaContext};
use half::bf16;
const INTERMEDIATE: usize = 17408;

pub struct Qwen38MlpBf16 {
    /// Raw HF post_attention_layernorm offset weight.
    pub norm: Tensor,
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
}
impl Qwen38MlpBf16 {
    /// Upload the four raw HF MLP tensors; GEMM weights are transposed to KN.
    pub fn from_hf_layer(ctx: &MusaContext, layer: usize, mut tensors: std::collections::HashMap<String, Tensor>) -> Result<Self> {
        if layer >= 64 { return Err(Error::Other("MLP layer index out of range".into())); }
        let mut take = |suffix: &str, shape: &[usize], transpose: bool| -> Result<Tensor> {
            let key = format!("model.language_model.layers.{layer}.{suffix}");
            let tensor = tensors.remove(&key).ok_or_else(|| Error::Other(format!("missing {key}")))?;
            if tensor.dtype() != DType::BF16 || tensor.shape().dims() != shape {
                return Err(Error::Other(format!("{key}: expected BF16 {shape:?}")));
            }
            let tensor = if transpose {
                let source = tensor.as_bf16()?;
                let (rows, cols) = (shape[0], shape[1]);
                let mut result = vec![bf16::ZERO; source.len()];
                for r in 0..rows { for c in 0..cols { result[c*rows+r] = source[r*cols+c]; } }
                Tensor::from_bf16(vec![cols,rows], &result)?
            } else { tensor };
            transfers::to_device(&tensor,ctx.device_id())
        };
        Ok(Self {
            norm: take("post_attention_layernorm.weight", &[HIDDEN], false)?,
            gate: take("mlp.gate_proj.weight", &[INTERMEDIATE,HIDDEN], true)?,
            up: take("mlp.up_proj.weight", &[INTERMEDIATE,HIDDEN], true)?,
            down: take("mlp.down_proj.weight", &[HIDDEN,INTERMEDIATE], true)?,
        })
    }
    pub fn forward(&self, ctx: &MusaContext, input: &Tensor) -> Result<Tensor> {
        for (name,tensor,shape) in [
            ("input",input,vec![1,HIDDEN]),
            ("norm",&self.norm,vec![HIDDEN]),
            ("gate",&self.gate,vec![HIDDEN,INTERMEDIATE]),
            ("up",&self.up,vec![HIDDEN,INTERMEDIATE]),
            ("down",&self.down,vec![INTERMEDIATE,HIDDEN]),
        ] {
            if tensor.dtype()!=DType::BF16 || tensor.shape().dims()!=shape || tensor.device()!=Device::Musa(ctx.device_id()) {
                return Err(Error::Other(format!("Qwen3.8 MLP invalid {name}: expected BF16 {shape:?} on context device")));
            }
        }
        let mut normalized=ctx.allocate_output(Shape::new(vec![1,HIDDEN]),DType::BF16)?;
        kernels::qwen38::offset_rms_bf16(ctx,input,&self.norm,&mut normalized,EPSILON)?;
        let gate=gemm(ctx,&normalized,&self.gate,vec![1,INTERMEDIATE])?;
        let up=gemm(ctx,&normalized,&self.up,vec![1,INTERMEDIATE])?;
        // Single-token rows: row concatenation gives contiguous gate|up halves.
        let packed=ops::concat_rows(ctx,&gate,&up)?.reshape(vec![1,2*INTERMEDIATE])?;
        let mut activation=ctx.allocate_output(Shape::new(vec![1,INTERMEDIATE]),DType::BF16)?;
        kernels::qwen38::swiglu_bf16(ctx,&packed,&mut activation)?;
        let mut out=gemm(ctx,&activation,&self.down,vec![1,HIDDEN])?;
        kernels::qwen38::add_into(ctx,&mut out,input)?;
        Ok(out)
    }
}
