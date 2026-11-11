// qwen38 (Qwen3.8-27B) device kernels for the MUSA backend — M1 batch 1.
//
// Ground truth for every kernel's math is the CUDA-side f64 oracle in
// crates/apxinf-cuda-new/tests/qwen38_gdn.rs (recurrence, conv, l2-norm) and
// crates/apxinf-cuda-new/native/kernels/custom/gdn_ops.cu:227-229 (decay and
// beta formulas). The Rust contract tests in
// crates/apxinf-musa/tests/qwen38_contracts.rs restate those oracles.
//
// Layout note: sequence-tensor kernels are BATCH-FIRST [tokens, channels]
// rows, matching the repo's descriptor convention (see the SDPA probe in
// devlocal/qwen38-musa/phase0/probes/ for the transposition trap that
// motivated pinning this down in writing).

// ── helpers ───────────────────────────────────────────────────────────────

__device__ __forceinline__ float qwen38_softplus(float x) {
  // gdn_ops.cu:227-228: x > 20 ? x : log1pf(expf(x))
  return x > 20.0f ? x : log1pf(expf(x));
}

__device__ __forceinline__ float qwen38_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float qwen38_silu(float x) {
  return x / (1.0f + expf(-x));
}

// ── GDN: fused recurrent step ─────────────────────────────────────────────
//
// state   [v_heads, v_dim, k_dim] F32, read-modify-write in place
// q, k    [k_heads, k_dim] BF16        (single token)
// v       [v_heads, v_dim] BF16        (single token)
// decay, beta [v_heads] F32            (already exponentiated inputs? NO:
//         `decay` is the LOG-space value; the kernel applies exp, per
//         reference_step: head_decay = decay[head].exp())
// out     [v_heads, v_dim] BF16
//
// Math per v-head h (k_head = h / (v_heads/k_heads)):
//   for each v_index: s *= exp(decay_h); p = <s, k>; s += (v - p) * beta_h * k;
//   out = <s, q> / sqrt(k_dim)
__global__ void qwen38_gdn_recurrent_step_kernel(
    float* state, const __mt_bfloat16* q, const __mt_bfloat16* k,
    const __mt_bfloat16* v, const float* decay, const float* beta,
    __mt_bfloat16* out, int v_heads, int k_heads, int v_dim, int k_dim) {
  const int head = blockIdx.x;        // one block per v-head
  const int v_index = threadIdx.x;    // one thread per v_index
  if (v_index >= v_dim) return;
  const int group = v_heads / k_heads;
  const int k_head = head / group;
  const float head_decay = expf(decay[head]);
  const float head_beta = beta[head];
  const __mt_bfloat16* q_row = q + (size_t)k_head * k_dim;
  const __mt_bfloat16* k_row = k + (size_t)k_head * k_dim;
  float* s = state + (size_t)head * v_dim * k_dim + (size_t)v_index * k_dim;

  float predicted = 0.0f;
  for (int i = 0; i < k_dim; ++i) {
    s[i] *= head_decay;
    predicted += s[i] * __bfloat162float(k_row[i]);
  }
  const float delta =
      (__bfloat162float(v[(size_t)head * v_dim + v_index]) - predicted) * head_beta;
  float readout = 0.0f;
  for (int i = 0; i < k_dim; ++i) {
    s[i] += delta * __bfloat162float(k_row[i]);
    readout += s[i] * __bfloat162float(q_row[i]);
  }
  out[(size_t)head * v_dim + v_index] =
      __float2bfloat16(readout / sqrtf((float)k_dim));
}

// ── GDN: decay & beta (single token) ──────────────────────────────────────
//
// a, b, a_log, dt_bias [heads] BF16 -> decay, beta [heads] F32
// gdn_ops.cu:226-229: decay = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b)
__global__ void qwen38_gdn_decay_beta_step_kernel(
    const __mt_bfloat16* a, const __mt_bfloat16* b, const __mt_bfloat16* a_log,
    const __mt_bfloat16* dt_bias, float* decay, float* beta, int heads) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= heads) return;
  const float shifted = __bfloat162float(a[h]) + __bfloat162float(dt_bias[h]);
  decay[h] = -expf(__bfloat162float(a_log[h])) * qwen38_softplus(shifted);
  beta[h] = qwen38_sigmoid(__bfloat162float(b[h]));
}

// Sequence variant: a, b [tokens, heads] BF16 (batch-first rows) ->
// decay, beta [tokens, heads] F32.
__global__ void qwen38_gdn_decay_beta_sequence_kernel(
    const __mt_bfloat16* a, const __mt_bfloat16* b, const __mt_bfloat16* a_log,
    const __mt_bfloat16* dt_bias, float* decay, float* beta, int tokens,
    int heads) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * heads) return;
  const int h = index % heads;
  const float shifted =
      __bfloat162float(a[index]) + __bfloat162float(dt_bias[h]);
  decay[index] = -expf(__bfloat162float(a_log[h])) * qwen38_softplus(shifted);
  beta[index] = qwen38_sigmoid(__bfloat162float(b[index]));
}

// ── GDN: causal conv, whole sequence ──────────────────────────────────────
//
// input  [tokens, channels] BF16 (batch-first)
// weight [channels, width]  BF16 (HF conv1d [C, 1, W] viewed as [C, W])
// out    [tokens, channels] BF16
// Zero left-padding: out[t][c] = silu(sum_i w[c][i] * x[t - W + 1 + i][c]).
// The shift-register semantics are the causal_conv_window test's rotate_left.
__global__ void qwen38_gdn_causal_conv_forward_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* weight, __mt_bfloat16* out,
    int tokens, int channels, int width) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  float w[8];
  for (int i = 0; i < width && i < 8; ++i)
    w[i] = __bfloat162float(weight[(size_t)c * width + i]);
  for (int t = 0; t < tokens; ++t) {
    float acc = 0.0f;
    for (int i = 0; i < width && i < 8; ++i) {
      const int src = t - width + 1 + i;
      if (src >= 0) acc += w[i] * __bfloat162float(input[(size_t)src * channels + c]);
    }
    out[(size_t)t * channels + c] = __float2bfloat16(qwen38_silu(acc));
  }
}

// Decode variant: window [channels, width] F32 is the rolling state
// (rotate_left shift register); advances it in place with the new token and
// writes the gated output for this token only.
__global__ void qwen38_gdn_causal_conv_step_kernel(
    float* window, const __mt_bfloat16* input, const __mt_bfloat16* weight,
    __mt_bfloat16* out, int channels, int width) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  float* w_row = window + (size_t)c * width;
  for (int i = 0; i < width - 1; ++i) w_row[i] = w_row[i + 1];
  w_row[width - 1] = __bfloat162float(input[c]);
  float acc = 0.0f;
  for (int i = 0; i < width; ++i)
    acc += w_row[i] * __bfloat162float(weight[(size_t)c * width + i]);
  out[c] = __float2bfloat16(qwen38_silu(acc));
}

// ── GDN: per-head L2 normalization, in place ──────────────────────────────
__global__ void qwen38_gdn_l2_normalize_heads_kernel(__mt_bfloat16* data,
                                                     int heads, int head_dim,
                                                     float eps) {
  const int head = blockIdx.x;
  __mt_bfloat16* row = data + (size_t)head * head_dim;
  __shared__ float per_thread[1024];
  __shared__ float inverse;
  float square_sum = 0.0f;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const float value = __bfloat162float(row[d]);
    square_sum += value * value;
  }
  per_thread[threadIdx.x] = square_sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int t = 0; t < blockDim.x; ++t) sum += per_thread[t];
    inverse = rsqrtf(sum + eps);
  }
  __syncthreads();
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    row[d] = __float2bfloat16(__bfloat162float(row[d]) * inverse);
}

// ── trivial vector kernels ────────────────────────────────────────────────

// dst[i] += src[i] over numel BF16 elements (residual accumulate).
__global__ void qwen38_add_into_kernel(__mt_bfloat16* dst, const __mt_bfloat16* src,
                                       long long numel) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + __bfloat162float(src[i]));
}

// Fused q|gate projection split: fused [rows, 2 * cols] BF16 ->
// q [rows, cols], gate [rows, cols]. The gate occupies the SECOND half of
// each row (matches the CUDA split_query_and_gate contract: q_proj width is
// 2 * HEADS * HEAD_DIM = q then gate).
__global__ void qwen38_split_query_and_gate_kernel(const __mt_bfloat16* fused,
                                                   __mt_bfloat16* q,
                                                   __mt_bfloat16* gate,
                                                   long long rows, long long cols) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long numel = rows * cols;
  if (i >= numel) return;
  const long long row = i / cols;
  const long long col = i % cols;
  q[i] = fused[row * 2 * cols + col];
  gate[i] = fused[row * 2 * cols + cols + col];
}

// out[i] *= sigmoid(gate[i]) over numel BF16 elements.
__global__ void qwen38_apply_output_gate_kernel(__mt_bfloat16* out,
                                                const __mt_bfloat16* gate,
                                                long long numel) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel) return;
  out[i] = __float2bfloat16(__bfloat162float(out[i]) *
                            qwen38_sigmoid(__bfloat162float(gate[i])));
}


// ── qwen38 sectioned mRoPE ────────────────────────────────────────────────
//
// data [tokens, heads, head_dim] BF16 in place; positions [tokens, 3] I32.
// Half-split pairing (element i with i + rotary_dim/2; the ABI name's
// "interleaved" refers to the AXES, not element pairing). Axes cycle
// T/H/W round-robin over pairs within the section budgets — the Qwen3.5
// convention checked by qwen38_attention.rs::partial_mrope_selects_qwen35_
// interleaved_axes: pair%3==1 && pair < 3*section_h -> H (axis 1);
// pair%3==2 && pair < 3*section_w -> W (axis 2); else T (axis 0).
__global__ void qwen38_rope_kernel(__mt_bfloat16* data, const int* positions,
                                   int heads, int head_dim, int rotary_dim,
                                   float theta, int section_h, int section_w) {
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int half = rotary_dim / 2;
  const int pair = threadIdx.x;
  if (pair >= half) return;
  const float freq = powf(theta, -2.0f * (float)pair / (float)rotary_dim);
  int axis = 0;
  if (pair % 3 == 1 && pair < section_h * 3) axis = 1;
  else if (pair % 3 == 2 && pair < section_w * 3) axis = 2;
  const float angle = (float)positions[t * 3 + axis] * freq;
  const float ca = cosf(angle), sa = sinf(angle);
  __mt_bfloat16* row = data + ((size_t)t * heads + h) * head_dim;
  const float low = __bfloat162float(row[pair]);
  const float high = __bfloat162float(row[pair + half]);
  row[pair] = __float2bfloat16(low * ca - high * sa);
  row[pair + half] = __float2bfloat16(high * ca + low * sa);
}
