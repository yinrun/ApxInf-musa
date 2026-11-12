//! Sequence/token parity and prefill-to-decode state contracts on MUSA.
use apxinf_core::{DType, Shape, Tensor};
use apxinf_musa::{kernels::qwen38, MusaBuffer, MusaContext};

fn upload(ctx: &MusaContext, values: &[f32], dims: &[usize], dtype: DType) -> (MusaBuffer, Tensor) {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|&v| match dtype {
            DType::BF16 => half::bf16::from_f32(v).to_bits().to_le_bytes().to_vec(),
            DType::F32 => v.to_le_bytes().to_vec(),
            _ => unreachable!(),
        })
        .collect();
    let buffer = MusaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    let tensor = buffer.as_tensor(Shape::new(dims.to_vec()), dtype).unwrap();
    (buffer, tensor)
}

fn read(buffer: &MusaBuffer, dtype: DType) -> Vec<f32> {
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    match dtype {
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        _ => unreachable!(),
    }
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "index {i}: actual {a}, expected {e}"
        );
    }
}

#[test]
fn decay_beta_sequence_matches_individual_tokens() {
    let ctx = MusaContext::new(0).unwrap();
    for (tokens, heads) in [(1, 1), (3, 7), (17, 48)] {
        let a: Vec<f32> = (0..tokens * heads)
            .map(|i| ((i * 13 % 53) as f32 - 26.0) / 4.0)
            .collect();
        let b: Vec<f32> = (0..tokens * heads)
            .map(|i| ((i * 7 % 41) as f32 - 20.0) / 3.0)
            .collect();
        let logs: Vec<f32> = (0..heads).map(|h| (h % 9) as f32 / 8.0 - 0.5).collect();
        let bias: Vec<f32> = (0..heads).map(|h| (h % 7) as f32 / 4.0 - 1.0).collect();
        let (_, at) = upload(&ctx, &a, &[tokens, heads], DType::BF16);
        let (_, bt) = upload(&ctx, &b, &[tokens, heads], DType::BF16);
        let (_, lt) = upload(&ctx, &logs, &[heads], DType::BF16);
        let (_, dt) = upload(&ctx, &bias, &[heads], DType::BF16);
        let (db, mut decay) = upload(
            &ctx,
            &vec![0.0; tokens * heads],
            &[tokens, heads],
            DType::F32,
        );
        let (bb, mut beta) = upload(
            &ctx,
            &vec![0.0; tokens * heads],
            &[tokens, heads],
            DType::F32,
        );
        qwen38::gdn_decay_beta_sequence(&ctx, &at, &bt, &lt, &dt, &mut decay, &mut beta).unwrap();
        ctx.synchronize().unwrap();
        let sequence_decay = read(&db, DType::F32);
        let sequence_beta = read(&bb, DType::F32);
        for t in 0..tokens {
            let (_, a) = upload(&ctx, &a[t * heads..(t + 1) * heads], &[heads], DType::BF16);
            let (_, b) = upload(&ctx, &b[t * heads..(t + 1) * heads], &[heads], DType::BF16);
            let (db, mut decay) = upload(&ctx, &vec![0.0; heads], &[heads], DType::F32);
            let (bb, mut beta) = upload(&ctx, &vec![0.0; heads], &[heads], DType::F32);
            qwen38::gdn_decay_beta_step(&ctx, &a, &b, &lt, &dt, &mut decay, &mut beta).unwrap();
            ctx.synchronize().unwrap();
            close(
                &read(&db, DType::F32),
                &sequence_decay[t * heads..(t + 1) * heads],
                1e-6,
            );
            close(
                &read(&bb, DType::F32),
                &sequence_beta[t * heads..(t + 1) * heads],
                1e-6,
            );
        }
    }
}

fn conv_case(tokens: usize, width: usize, channels: usize) {
    let ctx = MusaContext::new(0).unwrap();
    let raw: Vec<f32> = (0..(tokens + 1) * channels)
        .map(|i| ((i * 11 % 37) as f32 - 18.0) / 8.0)
        .collect();
    let weight: Vec<f32> = (0..channels * width)
        .map(|i| ((i * 3 % 13) as f32 - 6.0) / 8.0)
        .collect();
    let (_, input) = upload(
        &ctx,
        &raw[..tokens * channels],
        &[tokens, channels],
        DType::BF16,
    );
    let (_, weights) = upload(&ctx, &weight, &[channels, width], DType::BF16);
    let (ob, mut output) = upload(
        &ctx,
        &vec![0.0; tokens * channels],
        &[tokens, channels],
        DType::BF16,
    );
    // Dirty initial state must be replaced, including zero left padding.
    let (sb, mut state) = upload(
        &ctx,
        &vec![123.0; channels * width],
        &[channels, width],
        DType::F32,
    );
    qwen38::gdn_causal_conv_prefill(&ctx, &input, &weights, &mut state, &mut output).unwrap();
    ctx.synchronize().unwrap();
    let mut expected_state = vec![0.0; channels * width];
    let retained = tokens.min(width);
    for c in 0..channels {
        for j in 0..retained {
            expected_state[c * width + width - retained + j] =
                half::bf16::from_f32(raw[(tokens - retained + j) * channels + c]).to_f32();
        }
    }
    close(&read(&sb, DType::F32), &expected_state, 0.0);
    let (step_sb, mut step_state) = upload(
        &ctx,
        &vec![0.0; channels * width],
        &[channels, width],
        DType::F32,
    );
    let mut step_outputs = Vec::new();
    for t in 0..tokens {
        let (_, token) = upload(
            &ctx,
            &raw[t * channels..(t + 1) * channels],
            &[channels],
            DType::BF16,
        );
        let (b, mut out) = upload(&ctx, &vec![0.0; channels], &[channels], DType::BF16);
        qwen38::gdn_causal_conv_step(&ctx, &mut step_state, &token, &weights, &mut out).unwrap();
        ctx.synchronize().unwrap();
        step_outputs.extend(read(&b, DType::BF16));
    }
    close(&read(&ob, DType::BF16), &step_outputs, 0.0);
    close(&read(&sb, DType::F32), &read(&step_sb, DType::F32), 0.0);
    let (_, token) = upload(&ctx, &raw[tokens * channels..], &[channels], DType::BF16);
    let (db, mut decoded) = upload(&ctx, &vec![0.0; channels], &[channels], DType::BF16);
    qwen38::gdn_causal_conv_step(&ctx, &mut state, &token, &weights, &mut decoded).unwrap();
    let (_, whole_input) = upload(&ctx, &raw, &[tokens + 1, channels], DType::BF16);
    let (wb, mut whole_output) = upload(
        &ctx,
        &vec![0.0; (tokens + 1) * channels],
        &[tokens + 1, channels],
        DType::BF16,
    );
    qwen38::gdn_causal_conv_forward(&ctx, &whole_input, &weights, &mut whole_output).unwrap();
    ctx.synchronize().unwrap();
    close(
        &read(&db, DType::BF16),
        &read(&wb, DType::BF16)[tokens * channels..],
        0.0,
    );
}

#[test]
fn conv_short_prefill_handoff_matches_token_steps_and_whole_sequence() {
    for tokens in [1, 2, 3] {
        conv_case(tokens, 4, 7);
    }
}

#[test]
fn conv_long_prefill_handoff_matches_token_steps_and_whole_sequence() {
    for (tokens, width, channels) in [(4, 4, 7), (9, 4, 7), (17, 4, 48), (5, 1, 3)] {
        conv_case(tokens, width, channels);
    }
}
