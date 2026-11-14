//! W4 vision/language and unchanged BF16 expert/glue ownership.
use apxinf_core::{Result, Tensor};
use crate::musa_pi05::backend::{self, Context};
use super::*;
use super::bf16::modulation_to_device;

pub struct W4DeviceVisionBlock {
    pub norm1: Bf16DeviceLayerNorm,
    pub qkv: W4LinearWeights,
    pub output: W4LinearWeights,
    pub norm2: Bf16DeviceLayerNorm,
    pub fc1: W4LinearWeights,
    pub fc2: W4LinearWeights,
}

pub struct W4DeviceLanguageLayer {
    pub input_norm_scale: Tensor,
    pub qkv: Bf16LinearWeights,
    pub output: Bf16LinearWeights,
    pub post_attention_norm_scale: Tensor,
    pub gate_up: W4LinearWeights,
    pub down: W4LinearWeights,
}

pub struct W4Weights {
    pub patch_embedding: Bf16LinearWeights,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<W4DeviceVisionBlock>,
    pub vision_post_norm: Bf16DeviceLayerNorm,
    pub multimodal_projector: Bf16LinearWeights,
    pub token_embedding: Tensor,
    pub language_layers: Vec<W4DeviceLanguageLayer>,
    pub language_final_norm_scale: Tensor,
    pub action_layers: Vec<Bf16DeviceActionLayer>,
    pub action_final_modulation: Bf16LinearWeights,
    pub action_in: Bf16LinearWeights,
    pub action_out: Bf16LinearWeights,
    pub time_mlp_in: Bf16LinearWeights,
    pub time_mlp_out: Bf16LinearWeights,
}

impl W4Weights {
    /// Register the fixed weights the native assembly GEMM can take, and
    /// prepare the decode layouts it needs.
    ///
    /// Called once from `load`, before any traversal: registration is refused
    /// inside capture, and a weight left unregistered does not fail — it keeps
    /// the vendor path, which is the silent outcome this exists to prevent.
    ///
    /// NN registration retains the canonical `[input, output]` storage as it
    /// stands; NT also materialises a transposed copy. So this runs after the
    /// host checkpoint is dropped, which is what keeps the transposes from
    /// meeting a live FP32 copy of the model.
    pub fn prepare_assembly_weights(&self, context: &Context) -> Result<()> {
        for linear in [&self.action_in, &self.action_out] {
            backend::gemm::prepare_bf16_nt_weight(context, &linear.weight)?;
        }
        for layer in &self.action_layers {
            for linear in [&layer.qkv, &layer.output, &layer.down] {
                backend::gemm::prepare_bf16_nt_weight(context, &linear.weight)?;
            }
            // The concatenated `[gate | up]` stays in its stored layout.
            backend::gemm::prepare_bf16_nn_weight(context, &layer.gate_up.weight)?;
        }
        Ok(())
    }

    pub fn from_host(weights: &Pi05Weights, backend: &Context) -> Result<Self> {
        Ok(Self {
            patch_embedding: Bf16LinearWeights::from_host(
                &weights.vision.patch_embedding,
                backend,
            )?,
            position_embedding: bf16_to_device(&weights.vision.position_embedding, backend)?,
            vision_layers: weights
                .vision
                .blocks
                .iter()
                .map(|layer| W4DeviceVisionBlock::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            vision_post_norm: Bf16DeviceLayerNorm::from_host(
                &weights.vision.post_layer_norm,
                backend,
            )?,
            multimodal_projector: Bf16LinearWeights::from_host(
                &weights.vision.multimodal_projector,
                backend,
            )?,
            token_embedding: bf16_to_device(&weights.vision.token_embedding, backend)?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| W4DeviceLanguageLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            language_final_norm_scale: bf16_to_device(&weights.language_final_norm_scale, backend)?,
            action_layers: weights
                .action_layers
                .iter()
                .map(|layer| Bf16DeviceActionLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            action_final_modulation: modulation_to_device(&weights.action_final_norm, backend)?,
            action_in: Bf16LinearWeights::from_host(&weights.action_in, backend)?,
            action_out: Bf16LinearWeights::from_host(&weights.action_out, backend)?,
            time_mlp_in: Bf16LinearWeights::from_host(&weights.time_mlp_in, backend)?,
            time_mlp_out: Bf16LinearWeights::from_host(&weights.time_mlp_out, backend)?,
        })
    }
}

impl W4DeviceVisionBlock {
    pub(super) fn from_host(weights: &VisionBlockWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            norm1: Bf16DeviceLayerNorm::from_host(&weights.norm1, backend)?,
            qkv: W4LinearWeights::from_host_parts(
                &[&weights.q, &weights.k, &weights.v],
                backend,
            )?,
            output: W4LinearWeights::from_host(&weights.output, backend)?,
            norm2: Bf16DeviceLayerNorm::from_host(&weights.norm2, backend)?,
            fc1: W4LinearWeights::from_host(&weights.fc1, backend)?,
            fc2: W4LinearWeights::from_host(&weights.fc2, backend)?,
        })
    }
}

impl W4DeviceLanguageLayer {
    pub(super) fn from_host(weights: &LanguageLayerWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            input_norm_scale: bf16_to_device(&weights.input_norm_scale, backend)?,
            qkv: Bf16LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Bf16LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_norm_scale: bf16_to_device(&weights.post_attention_norm_scale, backend)?,
            gate_up: W4LinearWeights::from_host_parts_policy(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend, 16, 0.98,
            )?,
            down: W4LinearWeights::from_host_policy(&weights.mlp.down, backend, 16, 0.98)?,
        })
    }
}


pub struct W4LinearWeights {
    pub packed: backend::DeviceBuffer,
    pub scales: Tensor,
    pub bias: Option<Tensor>,
    pub k: usize,
    pub n: usize,
    pub group_size: usize,
    pub clip_ratio: f32,
}
impl W4LinearWeights {
    pub fn from_host(linear: &LinearWeights, ctx: &Context) -> Result<Self> { Self::from_host_policy(linear, ctx, 16, 0.98) }
    pub fn from_host_policy(linear: &LinearWeights, ctx: &Context, group_size: usize, clip_ratio: f32) -> Result<Self> { Self::from_host_parts_policy(&[linear], ctx, group_size, clip_ratio) }
    pub fn from_host_parts(parts: &[&LinearWeights], ctx: &Context) -> Result<Self> { Self::from_host_parts_policy(parts, ctx, 16, 0.98) }
    pub fn from_host_parts_policy(parts: &[&LinearWeights], ctx: &Context, group_size: usize, clip_ratio: f32) -> Result<Self> {
        let weight=super::packing::concat_host_2d(&parts.iter().map(|p| &p.weight).collect::<Vec<_>>())?;
        let dims=weight.shape().dims();
        let (k,n)=(dims[0],dims[1]);
        let (bytes,scales)=pack_w4(&weight.to_f32_vec()?,k,n,group_size,clip_ratio)?;
        let packed=backend::DeviceBuffer::alloc_on(ctx,bytes.len()).map_err(apxinf_core::Error::Other)?;
        packed.copy_from_host(&bytes).map_err(apxinf_core::Error::Other)?;
        let scales=backend::to_device(ctx,&Tensor::from_bf16(vec![n,k.div_ceil(group_size)],&scales)?)?;
        let bias=if parts.iter().all(|p| p.bias.is_none()) { None } else {
            let mut v=Vec::new();
            for p in parts { v.extend(p.bias.as_ref().ok_or_else(|| apxinf_core::Error::Other("mixed W4 bias presence".into()))?.to_f32_vec()?.into_iter().map(half::bf16::from_f32)); }
            Some(backend::to_device(ctx,&Tensor::from_bf16(vec![v.len()],&v)?)?)
        };
        Ok(Self { packed,scales,bias,k,n,group_size,clip_ratio })
    }
}

/// KN source -> NK bytes, low nibble first, BF16 scales per output/K128 group.
fn pack_w4(values: &[f32], k: usize, n: usize, group_size: usize, clip_ratio: f32) -> Result<(Vec<u8>,Vec<half::bf16>)> {
    use apxinf_core::Error;
    if k==0 || n==0 || group_size==0 || !clip_ratio.is_finite() || clip_ratio <= 0. || clip_ratio > 1. || k.checked_mul(n)!=Some(values.len()) || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Other("W4 requires finite nonempty KN weights".into()));
    }
    let groups=k.div_ceil(group_size);
    let kp=groups*group_size;
    let mut packed=vec![0u8;n*kp/2];
    let mut scales=vec![half::bf16::ONE;n*groups];
    for j in 0..n { for g in 0..groups {
        let end=((g+1)*group_size).min(k);
        let max=(g*group_size..end).map(|i| values[i*n+j].abs()).fold(0f32,f32::max);
        let scale=half::bf16::from_f32(if max==0. {1.} else {max*clip_ratio/7.});
        let sf=scale.to_f32();
        if !sf.is_normal() { return Err(Error::Other("W4 scale not representable as normal BF16".into())); }
        scales[j*groups+g]=scale;
        for i in g*group_size..end {
            let q=(values[i*n+j]/sf).round_ties_even().clamp(-7.,7.) as i8;
            packed[j*kp/2+i/2] |= ((q as u8)&15)<<((i%2)*4);
        }
    }}
    Ok((packed,scales))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stored_bf16_scale_controls_rounding() {
        let mut w=vec![0.;129];w[0]=1.;w[1]=0.2139;
        let (p,s)=pack_w4(&w,129,1,128,1.).unwrap();
        assert_eq!(s[0].to_bits(),0x3e12);
        assert_eq!(p[0],0x27);
        assert!(p[1..].iter().all(|v| *v==0));
        assert_eq!(s[1],half::bf16::ONE);
    }
    #[test]
    fn signed_nibbles_ties_even_and_padding() {
        let (p,_)=pack_w4(&[7.,0.5,1.5,2.5,-0.5,-1.5,-2.5],7,1,128,1.).unwrap();
        assert_eq!(&p[..4],&[0x07,0x22,0xe0,0x0e]);
        assert!(p[4..].iter().all(|v| *v==0));
        for v in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY,f32::from_bits(1)] {
            assert!(pack_w4(&[v],1,1,128,1.).is_err());
        }
    }
}

impl W4Weights {
    pub fn persistent_weight_bytes(&self) -> (usize,usize,usize) {
        fn tensor(t: &Tensor)->usize {t.shape().numel()*t.dtype().size_in_bytes()}
        fn bf(w: &Bf16LinearWeights)->usize {tensor(&w.weight)+w.bias.as_ref().map(tensor).unwrap_or(0)}
        fn norm(w:&Bf16DeviceLayerNorm)->usize {tensor(&w.weight)+tensor(&w.bias)}
        let mut packed=0; let mut scales=0;
        let mut other=bf(&self.patch_embedding)+tensor(&self.position_embedding)+norm(&self.vision_post_norm)+bf(&self.multimodal_projector)+tensor(&self.token_embedding)+tensor(&self.language_final_norm_scale);
        for l in &self.vision_layers {
            other+=norm(&l.norm1)+norm(&l.norm2);
            for w in [&l.qkv,&l.output,&l.fc1,&l.fc2] {packed+=w.packed.len();scales+=tensor(&w.scales);other+=w.bias.as_ref().map(tensor).unwrap_or(0);}
        }
        for l in &self.language_layers {
            other+=tensor(&l.input_norm_scale)+tensor(&l.post_attention_norm_scale);
            for w in [&l.gate_up,&l.down] { packed += w.packed.len(); scales += tensor(&w.scales); other += w.bias.as_ref().map(tensor).unwrap_or(0); }
            for w in [&l.qkv,&l.output] { other += bf(w); }
        }
        for l in &self.action_layers {for w in [&l.input_modulation,&l.qkv,&l.output,&l.post_attention_modulation,&l.gate_up,&l.down] {other+=bf(w);}}
        for w in [&self.action_final_modulation,&self.action_in,&self.action_out,&self.time_mlp_in,&self.time_mlp_out] {other+=bf(w);}
        (packed,scales,other)
    }
}
