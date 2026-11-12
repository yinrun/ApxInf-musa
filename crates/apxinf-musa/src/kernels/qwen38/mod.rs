//! qwen38 (Qwen3.8-27B) kernel facade — M1 batch 1.
//!
//! Device code: `kernels/custom/qwen38.cuh`; launchers: `adapters/custom_kernels.mu`
//! (`apxinf_musa_qwen38_*`). Math ground truth: the f64 oracles in
//! `crates/apxinf-cuda-new/tests/qwen38_gdn.rs` (recurrence, conv, l2-norm) and
//! `native/kernels/custom/gdn_ops.cu:227-229` (decay/beta formulas), restated by
//! `tests/qwen38_contracts.rs`.
//!
//! Sequence-tensor contracts are BATCH-FIRST `[tokens, ...]` rows — see the
//! module comment in the device file for why this is pinned in writing.

mod sequence;
pub use sequence::gdn_causal_conv_prefill;

use apxinf_core::{Error, Result, Tensor};

#[cfg(apxinf_musa_available)]
use super::contracts::{checked_i32, tensor_ptr};
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::ffi;
#[cfg(apxinf_musa_available)]
use crate::kernels::check_native;

fn require_shape(
    tensor: &Tensor,
    dtype: DType,
    want: &[usize],
    operation: &str,
    label: &str,
) -> Result<()> {
    let dims = tensor.shape().dims();
    if tensor.dtype() != dtype || dims != want {
        return Err(Error::Other(format!(
            "{operation}: {label} must be {dtype:?} {want:?}, got {:?} {dims:?}",
            tensor.dtype()
        )));
    }
    Ok(())
}

fn dims_of(tensor: &Tensor) -> Vec<usize> {
    tensor.shape().dims().to_vec()
}

fn numel_of(tensor: &Tensor) -> i64 {
    tensor.shape().dims().iter().product::<usize>() as i64
}

use apxinf_core::DType;

// ── GDN: fused recurrent step ─────────────────────────────────────────────

/// One GDN recurrence step over a single token. `state` is
/// `[v_heads, v_dim, k_dim]` F32, updated in place (log-space `decay` input;
/// the kernel applies `exp`). q/k `[k_heads, k_dim]`, v `[v_heads, v_dim]`
/// BF16; decay/beta `[v_heads]` F32; out `[v_heads, v_dim]` BF16.
pub fn gdn_recurrent_step(
    ctx: &MusaContext,
    state: &mut Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let sd = dims_of(state);
    if sd.len() != 3 {
        return Err(Error::Other(format!(
            "qwen38 gdn_recurrent_step: state must be 3-D, got {sd:?}"
        )));
    }
    let (v_heads, v_dim, k_dim) = (sd[0], sd[1], sd[2]);
    let k_rows = dims_of(q).first().copied().unwrap_or(0);
    if k_rows == 0 || v_dim == 0 || v_heads % k_rows != 0 {
        return Err(Error::Other(format!(
            "qwen38 gdn_recurrent_step: head sharing requires v_heads % k_heads == 0 \
             (v_heads={v_heads}, k_heads from q={k_rows})"
        )));
    }
    let k_heads = k_rows;
    require_shape(q, DType::BF16, &[k_heads, k_dim], "qwen38 gdn_recurrent_step", "q")?;
    require_shape(k, DType::BF16, &[k_heads, k_dim], "qwen38 gdn_recurrent_step", "k")?;
    require_shape(v, DType::BF16, &[v_heads, v_dim], "qwen38 gdn_recurrent_step", "v")?;
    require_shape(decay, DType::F32, &[v_heads], "qwen38 gdn_recurrent_step", "decay")?;
    require_shape(beta, DType::F32, &[v_heads], "qwen38 gdn_recurrent_step", "beta")?;
    require_shape(out, DType::BF16, &[v_heads, v_dim], "qwen38 gdn_recurrent_step", "out")?;
    if state.dtype() != DType::F32 {
        return Err(Error::Other(
            "qwen38 gdn_recurrent_step: state must be F32".into(),
        ));
    }
    launch_gdn_recurrent_step(ctx, state, q, k, v, decay, beta, out, v_heads, k_heads, v_dim, k_dim)
}

#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_gdn_recurrent_step(
    ctx: &MusaContext,
    state: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    out: &Tensor,
    v_heads: usize,
    k_heads: usize,
    v_dim: usize,
    k_dim: usize,
) -> Result<()> {
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_qwen38_gdn_recurrent_step(
                tensor_ptr(state)?,
                tensor_ptr(q)?,
                tensor_ptr(k)?,
                tensor_ptr(v)?,
                tensor_ptr(decay)?,
                tensor_ptr(beta)?,
                tensor_ptr(out)?,
                checked_i32("qwen38 gdn", "v_heads", v_heads)?,
                checked_i32("qwen38 gdn", "k_heads", k_heads)?,
                checked_i32("qwen38 gdn", "v_dim", v_dim)?,
                checked_i32("qwen38 gdn", "k_dim", k_dim)?,
                ctx.stream().handle(),
            ),
            "qwen38 gdn_recurrent_step",
        )
    }
}

// ── GDN: decay & beta ─────────────────────────────────────────────────────

/// `decay = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)` for one
/// token: a/b/a_log/dt_bias `[heads]` BF16 -> decay/beta `[heads]` F32.
pub fn gdn_decay_beta_step(
    ctx: &MusaContext,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    decay: &mut Tensor,
    beta: &mut Tensor,
) -> Result<()> {
    let heads = dims_of(a).first().copied().unwrap_or(0);
    require_shape(a, DType::BF16, &[heads], "qwen38 decay_beta", "a")?;
    require_shape(b, DType::BF16, &[heads], "qwen38 decay_beta", "b")?;
    require_shape(a_log, DType::BF16, &[heads], "qwen38 decay_beta", "a_log")?;
    require_shape(dt_bias, DType::BF16, &[heads], "qwen38 decay_beta", "dt_bias")?;
    require_shape(decay, DType::F32, &[heads], "qwen38 decay_beta", "decay")?;
    require_shape(beta, DType::F32, &[heads], "qwen38 decay_beta", "beta")?;
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_qwen38_gdn_decay_beta_step(
                tensor_ptr(a)?,
                tensor_ptr(b)?,
                tensor_ptr(a_log)?,
                tensor_ptr(dt_bias)?,
                tensor_ptr(decay)?,
                tensor_ptr(beta)?,
                checked_i32("qwen38 decay_beta", "heads", heads)?,
                ctx.stream().handle(),
            ),
            "qwen38 gdn_decay_beta_step",
        )
    }
}

/// Sequence variant: a/b `[tokens, heads]` BF16 -> decay/beta `[tokens, heads]` F32.
pub fn gdn_decay_beta_sequence(
    ctx: &MusaContext,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    decay: &mut Tensor,
    beta: &mut Tensor,
) -> Result<()> {
    let ad = dims_of(a);
    if ad.len() != 2 {
        return Err(Error::Other(format!(
            "qwen38 decay_beta_sequence: a must be 2-D [tokens, heads], got {ad:?}"
        )));
    }
    let (tokens, heads) = (ad[0], ad[1]);
    require_shape(b, DType::BF16, &ad, "qwen38 decay_beta_seq", "b")?;
    require_shape(a_log, DType::BF16, &[heads], "qwen38 decay_beta_seq", "a_log")?;
    require_shape(dt_bias, DType::BF16, &[heads], "qwen38 decay_beta_seq", "dt_bias")?;
    require_shape(decay, DType::F32, &ad, "qwen38 decay_beta_seq", "decay")?;
    require_shape(beta, DType::F32, &ad, "qwen38 decay_beta_seq", "beta")?;
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_qwen38_gdn_decay_beta_sequence(
                tensor_ptr(a)?,
                tensor_ptr(b)?,
                tensor_ptr(a_log)?,
                tensor_ptr(dt_bias)?,
                tensor_ptr(decay)?,
                tensor_ptr(beta)?,
                checked_i32("qwen38 decay_beta_seq", "tokens", tokens)?,
                checked_i32("qwen38 decay_beta_seq", "heads", heads)?,
                ctx.stream().handle(),
            ),
            "qwen38 gdn_decay_beta_sequence",
        )
    }
}

// ── GDN: causal conv ──────────────────────────────────────────────────────

/// Whole-sequence causal conv with zero left-padding and silu: input
/// `[tokens, channels]` BF16, weight `[channels, width]` BF16 (the HF
/// `conv1d.weight [C, 1, W]` viewed as `[C, W]`), out `[tokens, channels]` BF16.
pub fn gdn_causal_conv_forward(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let id = dims_of(input);
    let wd = dims_of(weight);
    if id.len() != 2 || wd.len() != 2 {
        return Err(Error::Other(format!(
            "qwen38 conv_forward: input must be [tokens, channels] and weight [channels, width], got {id:?} / {wd:?}"
        )));
    }
    let (tokens, channels) = (id[0], id[1]);
    let width = wd[1];
    require_shape(weight, DType::BF16, &[channels, width], "qwen38 conv_forward", "weight")?;
    require_shape(out, DType::BF16, &id, "qwen38 conv_forward", "out")?;
    require_shape(input, DType::BF16, &id, "qwen38 conv_forward", "input")?;
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_qwen38_gdn_causal_conv_forward(
                tensor_ptr(input)?,
                tensor_ptr(weight)?,
                tensor_ptr(out)?,
                checked_i32("qwen38 conv", "tokens", tokens)?,
                checked_i32("qwen38 conv", "channels", channels)?,
                checked_i32("qwen38 conv", "width", width)?,
                ctx.stream().handle(),
            ),
            "qwen38 gdn_causal_conv_forward",
        )
    }
}

/// Decode conv step: `window` `[channels, width]` F32 shift-register advanced
/// in place by one BF16 token; out `[channels]` BF16.
pub fn gdn_causal_conv_step(
    ctx: &MusaContext,
    window: &mut Tensor,
    input: &Tensor,
    weight: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let wd = dims_of(window);
    if wd.len() != 2 {
        return Err(Error::Other(format!(
            "qwen38 conv_step: window must be [channels, width], got {wd:?}"
        )));
    }
    let (channels, width) = (wd[0], wd[1]);
    if window.dtype() != DType::F32 {
        return Err(Error::Other("qwen38 conv_step: window must be F32".into()));
    }
    require_shape(input, DType::BF16, &[channels], "qwen38 conv_step", "input")?;
    require_shape(weight, DType::BF16, &[channels, width], "qwen38 conv_step", "weight")?;
    require_shape(out, DType::BF16, &[channels], "qwen38 conv_step", "out")?;
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_qwen38_gdn_causal_conv_step(
                tensor_ptr(window)?,
                tensor_ptr(input)?,
                tensor_ptr(weight)?,
                tensor_ptr(out)?,
                checked_i32("qwen38 conv_step", "channels", channels)?,
                checked_i32("qwen38 conv_step", "width", width)?,
                ctx.stream().handle(),
            ),
            "qwen38 gdn_causal_conv_step",
        )
    }
}

// ── GDN: per-head L2 normalization ────────────────────────────────────────

/// L2-normalize each head in place: `[heads, head_dim]` BF16.
pub fn gdn_l2_normalize_heads(ctx: &MusaContext, data: &mut Tensor, eps: f32) -> Result<()> {
    let dd = dims_of(data);
    if dd.len() != 2 {
        return Err(Error::Other(format!(
            "qwen38 l2_normalize: data must be [heads, head_dim], got {dd:?}"
        )));
    }
    require_shape(data, DType::BF16, &dd, "qwen38 l2_normalize", "data")?;
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_qwen38_gdn_l2_normalize_heads(
                tensor_ptr(data)?,
                checked_i32("qwen38 l2", "heads", dd[0])?,
                checked_i32("qwen38 l2", "head_dim", dd[1])?,
                eps,
                ctx.stream().handle(),
            ),
            "qwen38 gdn_l2_normalize_heads",
        )
    }
}

// ── trivial vector ops ────────────────────────────────────────────────────

/// `dst += src` elementwise, BF16, same shape.
pub fn add_into(ctx: &MusaContext, dst: &mut Tensor, src: &Tensor) -> Result<()> {
    let n = numel_of(dst);
    require_shape(src, DType::BF16, &dims_of(dst), "qwen38 add_into", "src")?;
    require_shape(dst, DType::BF16, &dims_of(dst), "qwen38 add_into", "dst")?;
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_qwen38_add_into(
                tensor_ptr(dst)?,
                tensor_ptr(src)?,
                n,
                ctx.stream().handle(),
            ),
            "qwen38 add_into",
        )
    }
}

/// Split fused `[rows, 2*cols]` (q then gate) into `q`/`gate` `[rows, cols]` BF16.
pub fn split_query_and_gate(
    ctx: &MusaContext,
    fused: &Tensor,
    q: &mut Tensor,
    gate: &mut Tensor,
) -> Result<()> {
    let fd = dims_of(fused);
    if fd.len() != 2 || fd[1] % 2 != 0 {
        return Err(Error::Other(format!(
            "qwen38 split_query_and_gate: fused must be [rows, 2*cols], got {fd:?}"
        )));
    }
    let (rows, cols) = (fd[0], fd[1] / 2);
    require_shape(q, DType::BF16, &[rows, cols], "qwen38 split", "q")?;
    require_shape(gate, DType::BF16, &[rows, cols], "qwen38 split", "gate")?;
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_qwen38_split_query_and_gate(
                tensor_ptr(fused)?,
                tensor_ptr(q)?,
                tensor_ptr(gate)?,
                rows as i64,
                cols as i64,
                ctx.stream().handle(),
            ),
            "qwen38 split_query_and_gate",
        )
    }
}

/// `out *= sigmoid(gate)` elementwise, BF16, same shape.
pub fn apply_output_gate(ctx: &MusaContext, out: &mut Tensor, gate: &Tensor) -> Result<()> {
    let n = numel_of(out);
    require_shape(gate, DType::BF16, &dims_of(out), "qwen38 gate", "gate")?;
    require_shape(out, DType::BF16, &dims_of(out), "qwen38 gate", "out")?;
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_qwen38_apply_output_gate(
                tensor_ptr(out)?,
                tensor_ptr(gate)?,
                n,
                ctx.stream().handle(),
            ),
            "qwen38 apply_output_gate",
        )
    }
}


/// BF16 GQA SDPA through muDNN. Rust tensors are batch-first contiguous:
/// Q/output `[batch, q_tokens, heads, 256]`, K/V
/// `[batch, k_tokens, kv_heads, 256]`. Prefill uses causal mode only when
/// `q_tokens == k_tokens`; decode uses non-causal mode because all KV-cache
/// entries are historical and muDNN's causal q!=k alignment is top-left.
pub fn sdpa_bf16(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &mut Tensor,
    causal: bool,
) -> Result<()> {
    let qd = dims_of(q);
    let kd = dims_of(k);
    if qd.len() != 4 || kd.len() != 4 {
        return Err(Error::Other(format!(
            "qwen38 sdpa: q/k must be 4-D [batch, seq, heads, 256], got {qd:?}/{kd:?}"
        )));
    }
    let batch = qd[0];
    let q_tokens = qd[1];
    let heads = qd[2];
    let dim = qd[3];
    let (k_batch, k_tokens, kv_heads, k_dim) = (kd[0], kd[1], kd[2], kd[3]);
    if batch == 0 || q_tokens == 0 || k_tokens == 0 {
        return Err(Error::Other("qwen38 sdpa: batch and sequences must be non-empty".into()));
    }
    if dim != 256 || k_dim != 256 || batch != k_batch || heads == 0 || kv_heads == 0
        || heads % kv_heads != 0
    {
        return Err(Error::Other(format!(
            "qwen38 sdpa: invalid GQA geometry q={qd:?}, k={kd:?}"
        )));
    }
    if causal && q_tokens != k_tokens {
        return Err(Error::Other(
            "qwen38 sdpa: causal mode requires q_tokens == k_tokens; decode is non-causal".into(),
        ));
    }
    for (name, tensor) in [("q", q), ("k", k), ("v", v), ("out", &*out)] {
        super::contracts::require_device(ctx, tensor, "qwen38 sdpa", name)?;
    }
    require_shape(q, DType::BF16, &qd, "qwen38 sdpa", "q")?;
    require_shape(k, DType::BF16, &kd, "qwen38 sdpa", "k")?;
    require_shape(v, DType::BF16, &kd, "qwen38 sdpa", "v")?;
    require_shape(out, DType::BF16, &[batch, q_tokens, heads, dim], "qwen38 sdpa", "out")?;
    unsafe {
        ffi::custom::check_mudnn(
            ffi::custom::apxinf_musa_qwen38_mudnn_sdpa_bf16(
                tensor_ptr(q)?,
                tensor_ptr(k)?,
                tensor_ptr(v)?,
                tensor_ptr(out)?,
                checked_i32("qwen38 sdpa", "batch", batch)?,
                checked_i32("qwen38 sdpa", "heads", heads)?,
                checked_i32("qwen38 sdpa", "kv_heads", kv_heads)?,
                checked_i32("qwen38 sdpa", "q_tokens", q_tokens)?,
                checked_i32("qwen38 sdpa", "k_tokens", k_tokens)?,
                checked_i32("qwen38 sdpa", "dim", dim)?,
                i32::from(causal),
                ctx.stream().handle(),
            ),
            || "qwen38 sdpa".to_string(),
        ).map_err(crate::error::musa_error)
    }
}

/// Apply partial half-split mRoPE with interleaved position axes.
pub fn rope(
    ctx: &MusaContext,
    data: &mut Tensor,
    positions: &Tensor,
    rotary_dim: usize,
    sections: [usize; 3],
    theta: f32,
) -> Result<()> {
    let dd = dims_of(data);
    if dd.len() != 3 {
        return Err(Error::Other(format!(
            "qwen38 rope: data must be [tokens, heads, head_dim], got {dd:?}"
        )));
    }
    let (tokens, heads, head_dim) = (dd[0], dd[1], dd[2]);
    if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Err(Error::Other(format!(
            "qwen38 rope: rotary_dim {rotary_dim} invalid for head_dim {head_dim}"
        )));
    }
    if sections.iter().sum::<usize>() != rotary_dim / 2 {
        return Err(Error::Other(format!(
            "qwen38 rope: sections {sections:?} must sum to rotary_dim/2 ({})",
            rotary_dim / 2
        )));
    }
    require_shape(data, DType::BF16, &dd, "qwen38 rope", "data")?;
    require_shape(positions, DType::I32, &[tokens, 3], "qwen38 rope", "positions")?;
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_qwen38_rope(
                tensor_ptr(data)?,
                tensor_ptr(positions)?,
                checked_i32("qwen38 rope", "tokens", tokens)?,
                checked_i32("qwen38 rope", "heads", heads)?,
                checked_i32("qwen38 rope", "head_dim", head_dim)?,
                checked_i32("qwen38 rope", "rotary_dim", rotary_dim)?,
                theta,
                checked_i32("qwen38 rope", "section_t", sections[0])?,
                checked_i32("qwen38 rope", "section_h", sections[1])?,
                checked_i32("qwen38 rope", "section_w", sections[2])?,
                ctx.stream().handle(),
            ),
            "qwen38 rope",
        )
    }
}
