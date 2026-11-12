//! Host-side checkpoint census and BF16 layout preparation.
use std::collections::HashMap;
use apxinf_core::{DType, Error, Result, Tensor};
use half::bf16;
use super::config::*;

#[derive(Debug)]
pub struct PlainBf16 { pub tensor: Tensor }
#[derive(Debug)]
pub struct GdnWeights { pub tensors: HashMap<String, PlainBf16> }
#[derive(Debug)]
pub struct FullAttentionWeights { pub tensors: HashMap<String, PlainBf16> }
#[derive(Debug)]
pub struct Qwen38Layer { pub gdn: Option<GdnWeights>, pub attention: Option<FullAttentionWeights>, pub tensors: HashMap<String, PlainBf16> }
#[derive(Debug)]
pub struct Qwen38Weights { pub root: HashMap<String, PlainBf16>, pub layers: Vec<Qwen38Layer> }

fn transpose_bf16(t: &Tensor) -> Result<Tensor> {
    if t.ndim() != 2 { return Err(Error::Other(format!("GEMM tensor must be 2-D, got {:?}", t.shape().dims()))); }
    let d = t.shape().dims();
    let src = t.as_bf16()?;
    let mut out = vec![bf16::from_f32(0.0); src.len()];
    for r in 0..d[0] { for c in 0..d[1] { out[c * d[0] + r] = src[r * d[1] + c]; } }
    Tensor::from_bf16(vec![d[1], d[0]], &out)
}
fn expected(name: &str) -> Option<Vec<usize>> {
    if name.ends_with("embed_tokens.weight") || name.ends_with("lm_head.weight") { Some(vec![VOCAB, HIDDEN]) }
    else if name.ends_with("input_layernorm.weight") || name.ends_with("post_attention_layernorm.weight") || name.ends_with("norm.weight") { Some(vec![HIDDEN]) }
    else if name.contains("gate_proj.weight") || name.contains("up_proj.weight") { Some(vec![INTERMEDIATE, HIDDEN]) }
    else if name.contains("down_proj.weight") { Some(vec![HIDDEN, INTERMEDIATE]) }
    else if name.contains("q_proj.weight") { Some(vec![HEADS * HEAD_DIM, HIDDEN]) }
    else if name.contains("k_proj.weight") { Some(vec![KV_HEADS * HEAD_DIM, HIDDEN]) }
    else if name.contains("v_proj.weight") { Some(vec![KV_HEADS * HEAD_DIM, HIDDEN]) }
    else if name.contains("o_proj.weight") { Some(vec![HIDDEN, HEADS * HEAD_DIM]) }
    else if name.ends_with("linear_attn.in_proj_qkv") { Some(vec![GDN_QKV, HIDDEN]) }
    else if name.ends_with("linear_attn.in_proj_z") { Some(vec![GDN_Z, HIDDEN]) }
    else if name.ends_with("linear_attn.out_proj") { Some(vec![HIDDEN, GDN_Z]) }
    else if name.ends_with("linear_attn.in_proj_a.weight") || name.ends_with("linear_attn.in_proj_b.weight") { Some(vec![GDN_A_B, HIDDEN]) }
    else if name.ends_with("linear_attn.conv1d.weight") { Some(vec![GDN_QKV, 1, GDN_CONV]) }
    else if name.ends_with("linear_attn.dt_bias") || name.ends_with("linear_attn.A_log") { Some(vec![GDN_V_HEADS]) }
    else if name.ends_with("linear_attn.norm.weight") { Some(vec![GDN_HEAD_DIM]) }
    else { None }
}
fn is_gemm(name: &str) -> bool { name.ends_with(".weight") && !name.ends_with("embed_tokens.weight") && !name.ends_with("lm_head.weight") && !name.ends_with("norm.weight") && !name.ends_with("layernorm.weight") && !name.ends_with("conv1d.weight") }

pub fn from_hf_map(input: HashMap<String, Tensor>) -> Result<Qwen38Weights> {
    let mut root = HashMap::new(); let mut layers: Vec<Qwen38Layer> = (0..LAYERS).map(|_| Qwen38Layer { gdn: None, attention: None, tensors: HashMap::new() }).collect();
    for (name, tensor) in input {
        if name.starts_with("model.visual.") || name.starts_with("model.mtp.") || name.starts_with("visual.") || name.starts_with("mtp.") { continue; }
        let short = name.strip_prefix("model.language_model.").ok_or_else(|| Error::Other(format!("unsupported Qwen3.8 tensor prefix: {name}")))?;
        let shape = tensor.shape().dims().to_vec();
        if let Some(rest) = short.strip_prefix("layers.") {
            let (idx, rest) = rest.split_once('.').ok_or_else(|| Error::Other(format!("bad layer tensor name {name}")))?;
            let layer: usize = idx.parse().map_err(|_| Error::Other(format!("bad layer index {name}")))?;
            if layer >= LAYERS { return Err(Error::Other(format!("layer {layer} out of range"))); }
            if let Some(want) = expected(rest) {
                if shape != want { return Err(Error::Other(format!("{name} shape {shape:?}, expected {want:?}"))); }
            } else { return Err(Error::Other(format!("unsupported Qwen3.8 text tensor {name}"))); }
            let tensor = if is_gemm(rest) { transpose_bf16(&tensor)? } else { if tensor.dtype()!=DType::BF16 { return Err(Error::Other(format!("{name} is {:?}, expected BF16", tensor.dtype()))); } tensor };
            layers[layer].tensors.insert(rest.to_string(), PlainBf16 { tensor });
        } else {
            if let Some(want) = expected(short) { if shape != want { return Err(Error::Other(format!("{name} shape {shape:?}, expected {want:?}"))); } }
            else { return Err(Error::Other(format!("unsupported Qwen3.8 root tensor {name}"))); }
            if tensor.dtype()!=DType::BF16 { return Err(Error::Other(format!("{name} is {:?}, expected BF16", tensor.dtype()))); }
            root.insert(short.to_string(), PlainBf16 { tensor });
        }
    }
    for (i, layer) in layers.iter_mut().enumerate() { if is_full_attention(i) { layer.attention=Some(FullAttentionWeights { tensors: std::mem::take(&mut layer.tensors) }); } else { layer.gdn=Some(GdnWeights { tensors: std::mem::take(&mut layer.tensors) }); } }
    Ok(Qwen38Weights { root, layers })
}

#[cfg(test)]
mod tests {
 use super::*;
 fn t(shape: Vec<usize>) -> Tensor { Tensor::from_bf16(shape.clone(), &vec![bf16::from_f32(0.0); shape.iter().product()]).unwrap() }
 #[test] fn maps_representative_hf_names_and_transposes() {
  let mut m=HashMap::new(); m.insert("model.language_model.embed_tokens.weight".into(),t(vec![VOCAB,HIDDEN])); m.insert("model.language_model.layers.0.mlp.gate_proj.weight".into(),t(vec![INTERMEDIATE,HIDDEN])); m.insert("model.visual.patch_embed.proj.weight".into(),t(vec![1])); m.insert("model.mtp.layers.0.weight".into(),t(vec![1]));
  let w=from_hf_map(m).unwrap(); assert_eq!(w.layers[0].gdn.as_ref().unwrap().tensors["mlp.gate_proj.weight"].tensor.shape().dims(), &[HIDDEN,INTERMEDIATE]); assert!(w.root.contains_key("embed_tokens.weight"));
 }
}
