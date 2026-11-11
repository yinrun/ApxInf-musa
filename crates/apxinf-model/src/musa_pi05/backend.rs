//! PI0.5's direct MUSA surface.
//!
//! The model tree reaches the device only through this seam: the same typed
//! operator contracts `apxinf-cuda-new` defines, the same session, the same
//! capture entry point. What differs is the runtime underneath, and the three
//! names `apxinf-musa` spells differently are translated here rather than in
//! the model copy.

use std::sync::Arc;

use apxinf_core::{DType, NormalGenerator, Result, Tensor};

pub(crate) use apxinf_musa::{
    ops, CapturedGraph, ExecutionSession, MusaBuffer as DeviceBuffer, MusaContext as Context,
    MusaDeviceCaps, MusaLibraryVersions, OperationShape,
};

pub(crate) use apxinf_musa::sampling;

/// The native assembly GEMM's two halves.
///
/// A weight is registered once, at load, and the scope marks a traversal as the
/// decode half of the schedule so the dispatcher may pick the decode tiles.
/// Registration is what makes them reachable at all: an unregistered weight
/// keeps the vendor path, silently.
pub(crate) mod gemm {
    pub(crate) use apxinf_musa::kernels::gemm::{
        adaptive_gate_residual_rms_quantize_w8a8_activation,
        adaptive_rms_quantize_w8a8_activation, decode_scope, norm_quantize_w8a8_activation,
        prepare_bf16_nn_weight, prepare_bf16_nt_weight, tilelang_serves_gemm,
    };
}

/// A normalized input prepared for its consuming projection.
///
/// The quantized variant carries the fused norm+quantize producer's rows and
/// scales; BF16 heads and excluded projections keep the materialized tensor.
pub enum NormalizedW8A8Input {
    Bf16(Tensor),
    Quantized(ops::w8a8::W8A8Activation),
}

/// Where this device's tactic measurements live, and how a session takes them.
pub(crate) mod tuning {
    pub(crate) use apxinf_musa::kernels::gemm::configure_tuning;
    pub(crate) use apxinf_musa::tuning::{TuningDb, TuningMode, TuningPaths};
}

/// Transfers, re-exported where the model tree calls them.
pub(crate) mod transfers {
    pub(crate) use apxinf_musa::transfers::{copy_cpu_to_device, to_cpu, to_device};
}

pub(crate) fn to_device(ctx: &Context, tensor: &Tensor) -> Result<Tensor> {
    transfers::to_device(tensor, ctx.device_id())
}

pub(crate) fn to_cpu(tensor: &Tensor) -> Result<Tensor> {
    transfers::to_cpu(tensor)
}

/// Apply an optional bias and elementwise activation in one typed pointwise
/// pass. Shared by the block path's SiLU/GELU sites and the weights fallback.
pub(crate) fn bias_activation_bf16(
    ctx: &Context,
    input: &Tensor,
    bias: Option<&Tensor>,
    activation: ops::PointwiseActivation,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    let mut args =
        ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, input, &mut output);
    args.bias = bias;
    args.activation = activation;
    ops::pointwise(ctx, args)?;
    Ok(output)
}

pub(crate) fn synchronize(ctx: &Context) -> Result<()> {
    ctx.synchronize()
}

/// Run `operation` with autotuning suppressed.
///
/// Warm-up and capture are the pair this exists for: a measurement inside
/// capture is illegal, and one on the warm-up's placeholder inputs would
/// persist a tactic chosen for a workload nobody runs.
pub(crate) fn without_autotune<T>(operation: impl FnOnce() -> T) -> T {
    apxinf_musa::tuning::without_autotune(operation)
}

pub(crate) fn create_normal_generator(
    ctx: &Context,
    output: Tensor,
) -> Result<Box<dyn NormalGenerator>> {
    sampling::create_normal_generator(ctx, output)
}

/// Memory layout of a fixed-shape batch of RGB `uint8` images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    Nhwc,
    Nchw,
}

impl std::fmt::Display for ImageLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nhwc => formatter.write_str("nhwc"),
            Self::Nchw => formatter.write_str("nchw"),
        }
    }
}

/// Capture a traversal the caller has already prepared on the same stream.
///
/// Takes `&Arc` because [`CapturedGraph`] keeps the context alive for the
/// executable's lifetime; the model tree holds its context in an `Arc` and
/// passes `&self.backend`.
pub(crate) fn capture(
    context: &Arc<Context>,
    operation: impl FnOnce() -> Result<()>,
) -> Result<CapturedGraph> {
    apxinf_musa::capture(Arc::clone(context), operation)
}
