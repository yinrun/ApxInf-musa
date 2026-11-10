//! Packed-QKV split and rotary position embedding contracts.

use apxinf_core::{Error, Result, Tensor};

use crate::kernels::attention as attention_kernels;
use crate::kernels::rope as rope_kernels;
use crate::MusaContext;

/// Which packed-QKV split the arguments describe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopeSemantic {
    /// Grouped-query split with rotary embedding on Q and K.
    SplitQkvRope,
    /// Vision MHA split with an optional bias and no rotation.
    SplitQkvBias,
}

/// Splits a packed QKV projection into separate Q, K and V buffers.
///
/// `k` and `v` may point either at fresh per-call buffers, which are
/// `[tokens, kv_heads, head_dim]`, or into a 2-D `[rows, head_dim]` KV cache,
/// in which case `kv_output_offset` is the row at which this call's tokens are
/// appended and the kernel writes in place. The buffers' rank selects the
/// form, not the offset: a prefix that writes a cache's first rows appends at
/// offset zero.
pub struct RopeArgs<'a> {
    pub semantic: RopeSemantic,
    pub qkv: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub q: &'a mut Tensor,
    pub k: &'a mut Tensor,
    pub v: &'a mut Tensor,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub theta: f32,
    pub position_offset: usize,
    pub kv_output_offset: usize,
}

pub fn rope(ctx: &MusaContext, args: RopeArgs<'_>) -> Result<()> {
    match args.semantic {
        RopeSemantic::SplitQkvRope => {
            // Rank, not the offset, picks the form: fresh per-call splits are
            // `[tokens, kv_heads, head_dim]`, while a KV cache is 2-D
            // `[rows, head_dim]` the kernel appends to at `kv_output_offset`.
            // A prefix writing a cache's first rows appends at offset zero, so
            // a zero offset alone cannot mean "fresh buffers".
            if args.k.shape().dims().len() == 2 {
                if args.kv_heads != 1 {
                    return Err(Error::Other(format!(
                        "MUSA rope cache write requires kv_heads == 1, got {}",
                        args.kv_heads
                    )));
                }
                // The cache case: the kernel writes K/V at the offset and
                // returns Q alone, so the borrowed k/v view is the cache
                // itself.
                rope_kernels::apply_q_write_kv_bf16_into(
                    ctx,
                    args.qkv,
                    args.bias,
                    args.q,
                    args.q_heads,
                    args.kv_heads,
                    args.head_dim,
                    args.theta,
                    args.position_offset,
                    args.k,
                    args.v,
                    args.kv_output_offset,
                )
            } else {
                if args.kv_output_offset != 0 {
                    return Err(Error::Other(
                        "MUSA rope fresh-buffer split has no KV-cache offset".into(),
                    ));
                }
                rope_kernels::split_qkv_apply_bf16_into(
                    ctx,
                    args.qkv,
                    args.bias,
                    args.q,
                    args.k,
                    args.v,
                    args.q_heads,
                    args.kv_heads,
                    args.head_dim,
                    args.theta,
                    args.position_offset,
                )
            }
        }
        RopeSemantic::SplitQkvBias => {
            if args.kv_output_offset != 0 {
                return Err(Error::Other(
                    "MUSA qkv split with a bias has no KV-cache form".into(),
                ));
            }
            attention_kernels::split_qkv_bias_bf16_into(
                ctx, args.qkv, args.bias, args.q, args.k, args.v, args.q_heads, args.head_dim,
            )
        }
    }
}
