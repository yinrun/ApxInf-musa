//! Correctness-first sequence helpers; device kernels remain in the parent module.
use super::{dims_of, gdn_causal_conv_forward, require_shape};
use crate::{MusaBuffer, MusaContext};
use apxinf_core::{DType, Error, Result, Tensor};

/// Zero-padded causal convolution followed by initialization of decode state.
///
/// `input`/`out` are BF16 `[tokens, channels]`, `weight` is BF16
/// `[channels, width]`, and `window` is F32 `[channels, width]`. The window is
/// replaced with the final raw input tokens, oldest first, with zero left
/// padding for short prompts. It never contains convolution/SiLU outputs.
///
/// This correctness-first handoff synchronizes the context stream, downloads
/// only the retained BF16 input tail, transposes/converts it on the host, and
/// uploads the F32 window using the existing buffer transfer operations. It is
/// not an asynchronous or graph-capture-compatible prefill implementation.
pub fn gdn_causal_conv_prefill(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    window: &mut Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let id = dims_of(input);
    let wd = dims_of(weight);
    if id.len() != 2 || wd.len() != 2 || id[0] == 0 || id[1] == 0 || wd[1] == 0 {
        return Err(Error::Other(format!(
            "qwen38 conv_prefill: expected nonempty input [tokens, channels] and weight [channels, width], got {id:?} / {wd:?}"
        )));
    }
    let (tokens, channels, width) = (id[0], id[1], wd[1]);
    require_shape(
        window,
        DType::F32,
        &[channels, width],
        "qwen38 conv_prefill",
        "window",
    )?;
    // Validate storage before launching the convolution or changing state.
    let input_buffer = MusaBuffer::from_tensor(input).map_err(Error::Other)?;
    let window_buffer = MusaBuffer::from_tensor(window).map_err(Error::Other)?;
    if input_buffer.device() != ctx.device_id() || window_buffer.device() != ctx.device_id() {
        return Err(Error::Other(
            "qwen38 conv_prefill: input/window device must match context".into(),
        ));
    }
    gdn_causal_conv_forward(ctx, input, weight, out)?;
    ctx.synchronize()?;
    let retained = tokens.min(width);
    let tail = input_buffer
        .view((tokens - retained) * channels * 2, retained * channels * 2)
        .map_err(Error::Other)?;
    let mut raw = vec![0u8; tail.len()];
    tail.copy_to_host(&mut raw).map_err(Error::Other)?;
    let mut state = vec![0u8; channels * width * 4];
    for t in 0..retained {
        for c in 0..channels {
            let src = (t * channels + c) * 2;
            let value = f32::from_bits((u16::from_le_bytes([raw[src], raw[src + 1]]) as u32) << 16);
            let dst = (c * width + width - retained + t) * 4;
            state[dst..dst + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    window_buffer.copy_from_host(&state).map_err(Error::Other)
}
