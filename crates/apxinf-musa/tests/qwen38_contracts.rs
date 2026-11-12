//! qwen38 MUSA kernel contract tests — M1 batch 1.
//!
//! Ports the hardware-independent f64 oracles from
//! `crates/apxinf-cuda-new/tests/qwen38_gdn.rs` (recurrence, causal conv,
//! l2-norm) plus a decay/beta formula check against the exact math at
//! `native/kernels/custom/gdn_ops.cu:227-229`. Tolerance 2e-2 as in the CUDA
//! suite (BF16 in f32 accumulation vs f64 reference over compounding steps).

use apxinf_core::{DType, Shape, Tensor};
use apxinf_musa::kernels::qwen38;
use apxinf_musa::{MusaBuffer, MusaContext};

fn bf16_bits(value: f32) -> u16 {
    // truncate-to-bf16: sufficient for oracle inputs (the CUDA suite's
    // `half::bf16::from_f32` is RNE; both round-trip through the same readback)
    let bits = value.to_bits();
    (bits >> 16) as u16
}

fn bf16_from_bits(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn upload_bf16(ctx: &MusaContext, values: &[f32], dims: Vec<usize>) -> (MusaBuffer, Tensor) {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| bf16_bits(*v).to_le_bytes())
        .collect();
    let buffer = MusaBuffer::alloc(bytes.len().max(1), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    let tensor = buffer.as_tensor(Shape::new(dims), DType::BF16).unwrap();
    (buffer, tensor)
}

fn upload_f32(ctx: &MusaContext, values: &[f32], dims: Vec<usize>) -> (MusaBuffer, Tensor) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = MusaBuffer::alloc(bytes.len().max(1), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    let tensor = buffer.as_tensor(Shape::new(dims), DType::F32).unwrap();
    (buffer, tensor)
}

fn read_bf16(buffer: &MusaBuffer) -> Vec<f32> {
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|b| bf16_from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

fn read_f32(buffer: &MusaBuffer) -> Vec<f32> {
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// The same recurrence the kernel implements, in f64 on the host (from
/// qwen38_gdn.rs `reference_step`).
#[allow(clippy::too_many_arguments)]
fn reference_step(
    state: &mut [f64],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: &[f32],
    beta: &[f32],
    v_heads: usize,
    k_heads: usize,
    v_dim: usize,
    k_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; v_heads * v_dim];
    let group = v_heads / k_heads;
    for head in 0..v_heads {
        let k_head = head / group;
        let head_decay = (decay[head] as f64).exp();
        let head_beta = beta[head] as f64;
        let base = head * v_dim * k_dim;
        for v_index in 0..v_dim {
            let row = base + v_index * k_dim;
            let mut predicted = 0.0f64;
            for index in 0..k_dim {
                state[row + index] *= head_decay;
                predicted += state[row + index] * k[k_head * k_dim + index] as f64;
            }
            let delta = (v[head * v_dim + v_index] as f64 - predicted) * head_beta;
            let mut out = 0.0f64;
            for index in 0..k_dim {
                state[row + index] += delta * k[k_head * k_dim + index] as f64;
                out += state[row + index] * q[k_head * k_dim + index] as f64;
            }
            output[head * v_dim + v_index] = (out / (k_dim as f64).sqrt()) as f32;
        }
    }
    output
}

#[test]
fn recurrent_step_matches_the_reference_over_many_tokens() {
    let ctx = MusaContext::new(0).unwrap();
    let (v_heads, k_heads, v_dim, k_dim) = (48usize, 16usize, 128usize, 128usize);
    let state_len = v_heads * v_dim * k_dim;

    let (state_buffer, mut state) = upload_f32(&ctx, &vec![0.0f32; state_len], vec![v_heads, v_dim, k_dim]);
    let (_out_buffer, mut output) = upload_bf16(&ctx, &vec![0.0f32; v_heads * v_dim], vec![v_heads, v_dim]);
    let mut host_state = vec![0.0f64; state_len];

    let mut worst = 0.0f64;
    for token in 0..6usize {
        let q: Vec<f32> = (0..k_heads * k_dim)
            .map(|i| (((i + token * 13) % 17) as f32 - 8.0) / 11.0)
            .collect();
        let k: Vec<f32> = (0..k_heads * k_dim)
            .map(|i| (((i * 3 + token * 7) % 19) as f32 - 9.0) / 13.0)
            .collect();
        let v: Vec<f32> = (0..v_heads * v_dim)
            .map(|i| (((i * 5 + token * 11) % 23) as f32 - 11.0) / 17.0)
            .collect();
        let decay: Vec<f32> = (0..v_heads).map(|h| -0.05 - 0.01 * (h % 7) as f32).collect();
        let beta: Vec<f32> = (0..v_heads).map(|h| 0.3 + 0.05 * (h % 5) as f32).collect();

        let (_qb, qt) = upload_bf16(&ctx, &q, vec![k_heads, k_dim]);
        let (_kb, kt) = upload_bf16(&ctx, &k, vec![k_heads, k_dim]);
        let (_vb, vt) = upload_bf16(&ctx, &v, vec![v_heads, v_dim]);
        let (_db, dt) = upload_f32(&ctx, &decay, vec![v_heads]);
        let (_bb, bt) = upload_f32(&ctx, &beta, vec![v_heads]);

        qwen38::gdn_recurrent_step(&ctx, &mut state, &qt, &kt, &vt, &dt, &bt, &mut output).unwrap();
        ctx.synchronize().unwrap();

        // reference consumes BF16-rounded inputs
        let q_rounded: Vec<f32> = q.iter().map(|x| bf16_from_bits(bf16_bits(*x))).collect();
        let k_rounded: Vec<f32> = k.iter().map(|x| bf16_from_bits(bf16_bits(*x))).collect();
        let v_rounded: Vec<f32> = v.iter().map(|x| bf16_from_bits(bf16_bits(*x))).collect();
        let expected = reference_step(
            &mut host_state, &q_rounded, &k_rounded, &v_rounded, &decay, &beta,
            v_heads, k_heads, v_dim, k_dim,
        );

        let produced = read_bf16(&_out_buffer);
        for (got, want) in produced.iter().zip(expected.iter()) {
            let denom = (want.abs() as f64).max(1e-2);
            worst = worst.max((*got as f64 - *want as f64).abs() / denom);
        }
    }
    println!("recurrent worst relative {worst:.5}");
    assert!(worst < 2e-2, "recurrent step drifted by {worst}");

    // state must match too
    let device_final = read_f32(&state_buffer);
    let mut state_worst = 0.0f64;
    for (got, want) in device_final.iter().zip(host_state.iter()) {
        let denom = want.abs().max(1e-2);
        state_worst = state_worst.max((*got as f64 - want).abs() / denom);
    }
    println!("state worst relative {state_worst:.5}");
    assert!(state_worst < 2e-2, "state drifted by {state_worst}");
}

#[test]
fn decay_beta_matches_the_formula() {
    let ctx = MusaContext::new(0).unwrap();
    let heads = 48usize;
    let a: Vec<f32> = (0..heads).map(|h| -0.4 + 0.01 * (h % 11) as f32).collect();
    let b: Vec<f32> = (0..heads).map(|h| -1.0 + 0.05 * (h % 9) as f32).collect();
    let a_log: Vec<f32> = (0..heads).map(|h| 0.02 * (h % 7) as f32).collect();
    let dt_bias: Vec<f32> = (0..heads).map(|h| 0.1 - 0.01 * (h % 5) as f32).collect();

    let (_ab, at) = upload_bf16(&ctx, &a, vec![heads]);
    let (_bb, bt) = upload_bf16(&ctx, &b, vec![heads]);
    let (_lb, lt) = upload_bf16(&ctx, &a_log, vec![heads]);
    let (_db, dtt) = upload_bf16(&ctx, &dt_bias, vec![heads]);
    let (decay_buffer, mut decay) = upload_f32(&ctx, &vec![0.0f32; heads], vec![heads]);
    let (beta_buffer, mut beta) = upload_f32(&ctx, &vec![0.0f32; heads], vec![heads]);

    qwen38::gdn_decay_beta_step(&ctx, &at, &bt, &lt, &dtt, &mut decay, &mut beta).unwrap();
    ctx.synchronize().unwrap();

    let decay_out = read_f32(&decay_buffer);
    let beta_out = read_f32(&beta_buffer);
    let softplus = |x: f64| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    for h in 0..heads {
        let a_r = bf16_from_bits(bf16_bits(a[h])) as f64;
        let b_r = bf16_from_bits(bf16_bits(b[h])) as f64;
        let al_r = bf16_from_bits(bf16_bits(a_log[h])) as f64;
        let dt_r = bf16_from_bits(bf16_bits(dt_bias[h])) as f64;
        let want_decay = -al_r.exp() * softplus(a_r + dt_r);
        let want_beta = 1.0 / (1.0 + (-b_r).exp());
        let d_err = ((decay_out[h] as f64 - want_decay) / want_decay.abs().max(1e-3)).abs();
        let b_err = (beta_out[h] as f64 - want_beta).abs();
        assert!(d_err < 2e-2, "decay[{h}]: {} vs {want_decay}", decay_out[h]);
        assert!(b_err < 2e-3, "beta[{h}]: {} vs {want_beta}", beta_out[h]);
    }
}

#[test]
fn causal_conv_window_advances_like_a_shift_register() {
    let ctx = MusaContext::new(0).unwrap();
    let (channels, width) = (64usize, 4usize);
    let (window_buffer, mut window) =
        upload_f32(&ctx, &vec![0.0f32; channels * width], vec![channels, width]);
    let weight_values: Vec<f32> = (0..channels * width)
        .map(|i| ((i % 5) as f32 - 2.0) / 4.0)
        .collect();
    let (_wb, wt) = upload_bf16(&ctx, &weight_values, vec![channels, width]);
    let (out_buffer, mut out) = upload_bf16(&ctx, &vec![0.0f32; channels], vec![channels]);

    let mut history = vec![vec![0.0f32; width]; channels];
    for token in 0..5usize {
        let input_values: Vec<f32> = (0..channels)
            .map(|c| ((c + token * 3) % 11) as f32 / 7.0 - 0.5)
            .collect();
        let (_ib, it) = upload_bf16(&ctx, &input_values, vec![channels]);
        qwen38::gdn_causal_conv_step(&ctx, &mut window, &it, &wt, &mut out).unwrap();
        ctx.synchronize().unwrap();

        let produced = read_bf16(&out_buffer);
        for c in 0..channels {
            history[c].rotate_left(1);
            history[c][width - 1] = bf16_from_bits(bf16_bits(input_values[c]));
            let mut acc = 0.0f32;
            for i in 0..width {
                acc += history[c][i] * bf16_from_bits(bf16_bits(weight_values[c * width + i]));
            }
            let expected = acc / (1.0 + (-acc).exp());
            let err = (produced[c] - expected).abs() / expected.abs().max(1e-2);
            assert!(err < 2e-2, "token {token} channel {c}: {} vs {expected}", produced[c]);
        }
    }
    let _ = &window_buffer;
}

#[test]
fn l2_normalization_makes_each_head_unit_norm() {
    let ctx = MusaContext::new(0).unwrap();
    let (heads, head_dim) = (16usize, 128usize);
    let values: Vec<f32> = (0..heads * head_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * (1.0 + (i / head_dim) as f32))
        .collect();
    let (buffer, mut data) = upload_bf16(&ctx, &values, vec![heads, head_dim]);
    qwen38::gdn_l2_normalize_heads(&ctx, &mut data, 1e-6).unwrap();
    ctx.synchronize().unwrap();

    let produced = read_bf16(&buffer);
    for head in 0..heads {
        let norm: f32 = produced[head * head_dim..(head + 1) * head_dim]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 2e-2, "head {head} norm {norm}");
    }
}

#[test]
fn add_into_and_gates_match_host_math() {
    let ctx = MusaContext::new(0).unwrap();
    let rows = 8usize;
    let cols = 32usize;
    let dst: Vec<f32> = (0..rows * cols).map(|i| ((i % 7) as f32 - 3.0) / 5.0).collect();
    let src: Vec<f32> = (0..rows * cols).map(|i| ((i % 11) as f32 - 5.0) / 9.0).collect();
    let fused: Vec<f32> = (0..rows * 2 * cols).map(|i| ((i % 13) as f32 - 6.0) / 7.0).collect();
    let gate: Vec<f32> = (0..rows * cols).map(|i| ((i % 9) as f32 - 4.0) / 6.0).collect();

    let round = |x: f32| bf16_from_bits(bf16_bits(x));

    // add_into
    let (dst_buffer, mut dst_t) = upload_bf16(&ctx, &dst, vec![rows, cols]);
    let (_sb, src_t) = upload_bf16(&ctx, &src, vec![rows, cols]);
    qwen38::add_into(&ctx, &mut dst_t, &src_t).unwrap();
    ctx.synchronize().unwrap();
    for (i, got) in read_bf16(&dst_buffer).iter().enumerate() {
        let want = round(round(dst[i]) + round(src[i]));
        let err = (got - want).abs() / want.abs().max(1e-2);
        assert!(err < 3e-2, "add_into[{i}]: {got} vs {want}");
    }

    // split_query_and_gate
    let (_fb, fused_t) = upload_bf16(&ctx, &fused, vec![rows, 2 * cols]);
    let (q_buffer, mut q_t) = upload_bf16(&ctx, &vec![0.0; rows * cols], vec![rows, cols]);
    let (g_buffer, mut g_t) = upload_bf16(&ctx, &vec![0.0; rows * cols], vec![rows, cols]);
    qwen38::split_query_and_gate(&ctx, &fused_t, &mut q_t, &mut g_t).unwrap();
    ctx.synchronize().unwrap();
    let qv = read_bf16(&q_buffer);
    let gv = read_bf16(&g_buffer);
    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(qv[r * cols + c], round(fused[r * 2 * cols + c]));
            assert_eq!(gv[r * cols + c], round(fused[r * 2 * cols + cols + c]));
        }
    }

    // apply_output_gate
    let (out_buffer, mut out_t) = upload_bf16(&ctx, &dst, vec![rows, cols]);
    let (_gb, gate_t) = upload_bf16(&ctx, &gate, vec![rows, cols]);
    qwen38::apply_output_gate(&ctx, &mut out_t, &gate_t).unwrap();
    ctx.synchronize().unwrap();
    for (i, got) in read_bf16(&out_buffer).iter().enumerate() {
        let g = round(gate[i]) as f64;
        let want = round(dst[i] as f32 * (1.0 / (1.0 + (-g).exp())) as f32);
        let err = (got - want).abs() / want.abs().max(1e-2);
        assert!(err < 3e-2, "gate[{i}]: {got} vs {want}");
    }
}


fn upload_i32(ctx: &MusaContext, values: &[i32], dims: Vec<usize>) -> (MusaBuffer, Tensor) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = MusaBuffer::alloc(bytes.len().max(1), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    let tensor = buffer.as_tensor(Shape::new(dims), DType::I32).unwrap();
    (buffer, tensor)
}

#[test]
fn rope_matches_the_qwen35_sectioned_oracle() {
    let ctx = MusaContext::new(0).unwrap();
    let head_dim = 256usize;
    let width = 64usize;
    let theta = 1.0e7f32;
    let sections = [11usize, 11, 10];
    let tokens = 2usize;
    let values: Vec<f32> = (0..tokens * 1 * head_dim)
        .map(|i| ((i % 29) as f32 - 14.0) / 9.0)
        .collect();
    let (buffer, mut data) = upload_bf16(&ctx, &values, vec![tokens, 1, head_dim]);
    let positions = [2i32, 17, 41, 3, 19, 43];
    let (_pb, pt) = upload_i32(&ctx, &positions, vec![tokens, 3]);

    qwen38::rope(&ctx, &mut data, &pt, width, sections, theta).unwrap();
    ctx.synchronize().unwrap();
    let produced = read_bf16(&buffer);

    let round = |x: f32| bf16_from_bits(bf16_bits(x));
    let half = width / 2;
    for t in 0..tokens {
        for pair in 0..half {
            let axis = if pair % 3 == 1 && pair < sections[1] * 3 {
                1
            } else if pair % 3 == 2 && pair < sections[2] * 3 {
                2
            } else {
                0
            };
            let angle = positions[t * 3 + axis] as f64
                * (theta as f64).powf(-2.0 * pair as f64 / width as f64);
            let base = t * head_dim;
            let low = round(values[base + pair]) as f64;
            let high = round(values[base + half + pair]) as f64;
            let want = [
                low * angle.cos() - high * angle.sin(),
                high * angle.cos() + low * angle.sin(),
            ];
            for (got, w) in [
                (produced[base + pair] as f64, want[0]),
                (produced[base + half + pair] as f64, want[1]),
            ] {
                assert!(
                    (got - w).abs() / w.abs().max(1e-2) < 2e-2,
                    "token {t} pair {pair} axis {axis}: got {got}, want {w}"
                );
            }
        }
        // past the rotary prefix untouched
        for d in width..head_dim {
            assert_eq!(produced[t * head_dim + d], round(values[t * head_dim + d]));
        }
    }
}

#[test]
fn rope_with_equal_triples_is_plain_partial_rope() {
    let ctx = MusaContext::new(0).unwrap();
    let (tokens, heads, head_dim) = (4usize, 4usize, 256usize);
    let width = 64usize;
    let theta = 1.0e7f32;
    let sections = [11usize, 11, 10];
    let values: Vec<f32> = (0..tokens * heads * head_dim)
        .map(|i| ((i % 31) as f32 - 15.0) / 8.0)
        .collect();
    let (buffer, mut data) = upload_bf16(&ctx, &values, vec![tokens, heads, head_dim]);
    let positions: Vec<i32> = (0..tokens as i32)
        .flat_map(|p| [p * 7, p * 7, p * 7])
        .collect();
    let (_pb, pt) = upload_i32(&ctx, &positions, vec![tokens, 3]);

    qwen38::rope(&ctx, &mut data, &pt, width, sections, theta).unwrap();
    ctx.synchronize().unwrap();
    let produced = read_bf16(&buffer);

    let round = |x: f32| bf16_from_bits(bf16_bits(x));
    let half = width / 2;
    for t in 0..tokens {
        let pos = (t as i32 * 7) as f64;
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            for i in 0..half {
                let angle = pos * (theta as f64).powf(-2.0 * i as f64 / width as f64);
                let low = round(values[base + i]) as f64;
                let high = round(values[base + half + i]) as f64;
                let want = [
                    low * angle.cos() - high * angle.sin(),
                    high * angle.cos() + low * angle.sin(),
                ];
                for (got, w) in [
                    (produced[base + i] as f64, want[0]),
                    (produced[base + half + i] as f64, want[1]),
                ] {
                    assert!(
                        (got - w).abs() / w.abs().max(1e-2) < 2e-2,
                        "t{t} h{h} i{i}: {got} vs {w}"
                    );
                }
            }
        }
    }
}

// Zero Q/K make softmax uniform. Distinct V values per token/head expose
// incorrect causal masking, GQA head mapping, and batch-first strides.
fn sdpa_uniform_case(q_len: usize, k_len: usize, causal: bool) {
    let ctx = MusaContext::new(0).unwrap();
    let (_, q) = upload_bf16(&ctx, &vec![0.; q_len * 24 * 256], vec![1,q_len,24,256]);
    let (_, k) = upload_bf16(&ctx, &vec![0.; k_len * 4 * 256], vec![1,k_len,4,256]);
    let values: Vec<f32> = (0..k_len*4*256).map(|i| {
        let t = i / (4*256); let h = (i/256)%4;
        (t + 2*h) as f32 / 16.
    }).collect();
    let (_, v) = upload_bf16(&ctx, &values, vec![1,k_len,4,256]);
    let (buffer, mut out) = upload_bf16(&ctx, &vec![-99.; q_len*24*256], vec![1,q_len,24,256]);
    // Repeat with the same shape to exercise cached descriptors and scratch.
    for _ in 0..2 {
        qwen38::sdpa_bf16(&ctx,&q,&k,&v,&mut out,causal).unwrap();
        ctx.synchronize().unwrap();
        for (i, got) in read_bf16(&buffer).iter().enumerate() {
            let t=i/(24*256); let h=(i/256)%24;
            let count=if causal {t+1} else {k_len};
            let want=((count-1) as f32/2. + 2.*(h/6) as f32)/16.;
            assert!(got.is_finite() && (got-want).abs()<0.016,
                "SDPA t={t} h={h}: {got} != {want}");
        }
    }
}

#[test]
fn sdpa_causal_prefill_uses_only_prefix_and_shares_kv_heads() {
    sdpa_uniform_case(16,16,true);
}

#[test]
fn sdpa_decode_reads_entire_valid_cache() {
    sdpa_uniform_case(1,32,false);
}

#[test]
fn sdpa_rejects_empty_sequence_before_vendor_dispatch() {
    let ctx = MusaContext::new(0).unwrap();
    let q = Tensor::zeros(vec![1,0,24,256], DType::BF16);
    let k = Tensor::zeros(vec![1,0,4,256], DType::BF16);
    let mut out = Tensor::zeros(vec![1,0,24,256], DType::BF16);
    let error=qwen38::sdpa_bf16(&ctx,&q,&k,&k,&mut out,true).unwrap_err();
    assert!(error.to_string().contains("non-empty"), "wrong validation boundary: {error}");
}

#[test]
fn sdpa_random_nonzero_matches_f64_softmax() {
    let ctx=MusaContext::new(0).unwrap();
    for (batch,ql,kl,causal) in [(1,17,17,true),(2,1,33,false),(1,1,129,false)] {
        let mut seed=713u64;
        let mut random=|n:usize| -> Vec<f32> {(0..n).map(|_| {
            seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            bf16_from_bits(bf16_bits(((seed>>32) as u32 as f64/u32::MAX as f64*2.-1.) as f32))
        }).collect()};
        let qv=random(batch*ql*24*256); let kv=random(batch*kl*4*256); let vv=random(batch*kl*4*256);
        let (_,q)=upload_bf16(&ctx,&qv,vec![batch,ql,24,256]);
        let (_,k)=upload_bf16(&ctx,&kv,vec![batch,kl,4,256]);
        let (_,v)=upload_bf16(&ctx,&vv,vec![batch,kl,4,256]);
        let (ob,mut out)=upload_bf16(&ctx,&vec![0.;qv.len()],vec![batch,ql,24,256]);
        qwen38::sdpa_bf16(&ctx,&q,&k,&v,&mut out,causal).unwrap();ctx.synchronize().unwrap();
        let got=read_bf16(&ob);let(mut err,mut norm)=(0f64,0f64);
        for b in 0..batch {for t in 0..ql {for h in 0..24 {
            let count=if causal {t+1}else{kl};
            let qi=((b*ql+t)*24+h)*256;
            let mut scores:Vec<f64>=(0..count).map(|j| {
                let ki=((b*kl+j)*4+h/6)*256;
                (0..256).map(|d|qv[qi+d] as f64*kv[ki+d] as f64).sum::<f64>()/16.
            }).collect();
            let max=scores.iter().copied().fold(f64::NEG_INFINITY,f64::max);
            for s in &mut scores {*s=(*s-max).exp();}
            let denom=scores.iter().sum::<f64>();
            for d in 0..256 {
                let want=(0..count).map(|j|scores[j]*vv[((b*kl+j)*4+h/6)*256+d] as f64).sum::<f64>()/denom;
                assert!(got[qi+d].is_finite());err+=(got[qi+d] as f64-want).powi(2);norm+=want*want;
            }
        }}}
        let rel=(err/norm).sqrt();println!("SDPA batch={batch} q={ql} k={kl} rel_l2={rel}");assert!(rel<0.008);
    }
}

#[test]
fn sdpa_rejects_invalid_geometry_dtype_and_causal_lengths() {
    let ctx=MusaContext::new(0).unwrap();
    let (_,q)=upload_bf16(&ctx,&vec![0.;24*256],vec![1,1,24,256]);
    let (_,k)=upload_bf16(&ctx,&vec![0.;2*4*256],vec![1,2,4,256]);
    let (_,mut out)=upload_bf16(&ctx,&vec![0.;24*256],vec![1,1,24,256]);
    assert!(qwen38::sdpa_bf16(&ctx,&q,&k,&k,&mut out,true).is_err());
    let (_,bad)=upload_f32(&ctx,&vec![0.;24*256],vec![1,1,24,256]);
    assert!(qwen38::sdpa_bf16(&ctx,&bad,&k,&k,&mut out,false).is_err());
    let (_,bad)=upload_bf16(&ctx,&vec![0.;2*5*256],vec![1,2,5,256]);
    assert!(qwen38::sdpa_bf16(&ctx,&q,&bad,&bad,&mut out,false).is_err());
    let cpu=Tensor::zeros(vec![1,1,24,256],DType::BF16);
    assert!(qwen38::sdpa_bf16(&ctx,&cpu,&k,&k,&mut out,false).is_err());
}

#[test]
fn sdpa_rejects_other_device_before_dispatch() {
    let ctx=MusaContext::new(0).unwrap();
    let other=MusaContext::new(1).unwrap();
    let (_,q)=upload_bf16(&other,&vec![0.;24*256],vec![1,1,24,256]);
    let (_,k)=upload_bf16(&ctx,&vec![0.;4*256],vec![1,1,4,256]);
    let (_,mut out)=upload_bf16(&ctx,&vec![0.;24*256],vec![1,1,24,256]);
    let result=qwen38::sdpa_bf16(&ctx,&q,&k,&k,&mut out,false);
    assert!(result.is_err(), "cross-device operand reached vendor dispatch");
}
