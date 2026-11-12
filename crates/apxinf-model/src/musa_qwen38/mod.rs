//! HF checkpoint inspection and an isolated eager GDN mixer; no full model runtime yet.
pub mod model;
pub mod request;
pub mod runner;
mod config;
mod weights;
pub(crate) use config::Qwen38Config;

use std::collections::HashMap;
use apxinf_core::{Device, Error, Result, Tensor};
use crate::auto::{LoadOptions, LoadedModel};
use std::path::Path;
use std::sync::Arc;
use apxinf_core::Backend;

pub(crate) fn load(path: &Path, device: Device, options: &LoadOptions) -> Result<LoadedModel> {
    let root = if path.is_dir() { path } else { path.parent().unwrap_or(Path::new(".")) };
    let _config = Qwen38Config::from_json_file(&root.join("config.json"))?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
        .map_err(|e| Error::Other(format!("load {}: {e}", path.display())))?;
    let _weights = weights::from_hf_map(tensors)?;
    let _ = (device, options);
    Err(Error::Other("MUSA Qwen3.8 weights are validated and loaded, but forward is not implemented yet".into()))
}

pub(crate) fn load_registered(path: &Path, device: Device, _backend: Arc<dyn Backend>, options: &LoadOptions) -> Result<LoadedModel> {
    load(path, device, options)
}

/// Validate an in-memory HF tensor map without constructing a runtime.
pub fn inspect_hf_weights(tensors: HashMap<String, Tensor>) -> Result<usize> {
    let weights = weights::from_hf_map(tensors)?;
    Ok(weights.root.len() + weights.layers.iter().map(|l| l.gdn.as_ref().map_or(0, |x| x.tensors.len()) + l.attention.as_ref().map_or(0, |x| x.tensors.len())).sum::<usize>())
}
