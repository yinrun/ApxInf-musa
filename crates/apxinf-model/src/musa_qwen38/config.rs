//! Fixed Qwen3.8-27B text configuration used by the MUSA weight inspector.

use apxinf_core::{Error, Result};
use apxinf_loader::ModelConfig;
use serde::Deserialize;
use std::{fs, path::Path};

pub const HIDDEN: usize = 5120;
pub const INTERMEDIATE: usize = 17408;
pub const VOCAB: usize = 248_320;
pub const LAYERS: usize = 64;
pub const HEADS: usize = 24;
pub const KV_HEADS: usize = 4;
pub const HEAD_DIM: usize = 256;
pub const GDN_QKV: usize = 10_240;
pub const GDN_Z: usize = 6_144;
pub const GDN_V_HEADS: usize = 48;
pub const GDN_HEAD_DIM: usize = 128;
pub const GDN_A_B: usize = 48;
pub const GDN_CONV: usize = 4;
pub const RMS_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 10_000_000.0;

#[derive(Clone, Debug)]
pub struct Qwen38Config {
    pub text: ModelConfig,
    pub full_attention_interval: usize,
}

#[derive(Deserialize)]
struct RootConfig {
    model_type: Option<String>,
    text_config: Option<TextConfig>,
}

#[derive(Deserialize)]
struct TextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
    max_position_embeddings: Option<usize>,
    rope_theta: Option<f32>,
    rms_norm_eps: Option<f32>,
    head_dim: Option<usize>,
    rope_parameters: Option<RopeParameters>,
}

#[derive(Deserialize)]
struct RopeParameters {
    rope_theta: Option<f32>,
}

impl Qwen38Config {
    pub fn fixed() -> Self {
        Self {
            text: ModelConfig {
                hidden_size: HIDDEN,
                intermediate_size: INTERMEDIATE,
                n_layers: LAYERS,
                n_heads: HEADS,
                n_kv_heads: KV_HEADS,
                vocab_size: VOCAB,
                max_seq_len: 32_768,
                rope_theta: ROPE_THETA,
                rms_norm_eps: RMS_EPS,
            },
            full_attention_interval: 4,
        }
    }

    pub fn from_json_file(path: &Path) -> Result<Self> {
        let value: RootConfig = serde_json::from_slice(
            &fs::read(path).map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?,
        )
        .map_err(|e| Error::Other(format!("parse {}: {e}", path.display())))?;
        let text = value.text_config.ok_or_else(|| Error::Other(
            "Qwen3.8 checkpoint config is missing nested text_config".into(),
        ))?;
        let expected = Self::fixed();
        let checks = [
            ("hidden_size", text.hidden_size, HIDDEN),
            ("intermediate_size", text.intermediate_size, INTERMEDIATE),
            ("num_hidden_layers", text.num_hidden_layers, LAYERS),
            ("num_attention_heads", text.num_attention_heads, HEADS),
            ("num_key_value_heads", text.num_key_value_heads, KV_HEADS),
            ("vocab_size", text.vocab_size, VOCAB),
        ];
        for (name, got, want) in checks {
            if got != want {
                return Err(Error::Other(format!("Qwen3.8 text_config {name}={got}, expected {want}")));
            }
        }
        if let Some(got) = text.head_dim {
            if got != HEAD_DIM { return Err(Error::Other(format!("text_config head_dim={got}, expected {HEAD_DIM}"))); }
        }
        if let Some(got) = text.rope_parameters.as_ref().and_then(|r| r.rope_theta) {
            if !got.is_finite() || (got - ROPE_THETA).abs() > 1.0 {
                return Err(Error::Other(format!("text_config rope_parameters.rope_theta={got}, expected {ROPE_THETA}")));
            }
        }
        if let Some(got) = text.rope_theta {
            if (got - ROPE_THETA).abs() > 1.0 { return Err(Error::Other(format!("text_config rope_theta={got}, expected {ROPE_THETA}"))); }
        }
        if let Some(got) = text.rms_norm_eps {
            if (got - RMS_EPS).abs() > 1e-9 { return Err(Error::Other(format!("text_config rms_norm_eps={got}, expected {RMS_EPS}"))); }
        }
        if !matches!(value.model_type.as_deref(), Some("qwen3_5" | "qwen3_5_text")) {
            return Err(Error::Other("expected HF qwen3_5 or qwen3_5_text model_type".into()));
        }
        Ok(expected)
    }
}

pub(crate) fn is_full_attention(layer: usize) -> bool { (layer + 1) % 4 == 0 }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_wrong_nested_rope_theta() {
        let value = serde_json::json!({"model_type":"qwen3_5", "text_config":{
            "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":64,
            "num_attention_heads":24,"num_key_value_heads":4,"vocab_size":248320,
            "head_dim":256,"rope_parameters":{"rope_theta":10000.0}
        }});
        let path=std::env::temp_dir().join(format!("qwen38-config-{}.json",std::process::id()));
        fs::write(&path,serde_json::to_vec(&value).unwrap()).unwrap();
        let result=Qwen38Config::from_json_file(&path);
        fs::remove_file(&path).unwrap();
        assert!(result.is_err(),"nested RoPE settings must not be silently ignored");
    }
    #[test]
    fn rejects_unrelated_model_type_with_matching_dimensions() {
        let value = serde_json::json!({"model_type":"llama", "text_config":{
            "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":64,
            "num_attention_heads":24,"num_key_value_heads":4,"vocab_size":248320
        }});
        let path=std::env::temp_dir().join(format!("qwen38-identity-{}.json",std::process::id()));
        fs::write(&path,serde_json::to_vec(&value).unwrap()).unwrap();
        let result=Qwen38Config::from_json_file(&path);
        fs::remove_file(&path).unwrap();
        assert!(result.is_err(),"unrelated model family must be rejected before weight loading");
    }

}
