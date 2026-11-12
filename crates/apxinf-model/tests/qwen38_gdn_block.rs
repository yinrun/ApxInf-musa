#![cfg(feature = "musa")]
use apxinf_core::{DType, Tensor};
use apxinf_model::musa_qwen38::model::{
    Qwen38GdnBf16Block, Qwen38GdnBf16State, Qwen38GdnBf16Weights,
};
use apxinf_musa::{transfers, MusaContext};
use half::bf16;
fn round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn upload(ctx: &MusaContext, dims: Vec<usize>, values: Vec<f32>) -> Tensor {
    transfers::to_device(
        &Tensor::from_bf16(
            dims,
            &values.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
        )
        .unwrap(),
        ctx.device_id(),
    )
    .unwrap()
}
fn constant(ctx: &MusaContext, dims: Vec<usize>, x: f32) -> Tensor {
    let n = dims.iter().product();
    upload(ctx, dims, vec![x; n])
}
fn projection(ctx: &MusaContext, n: usize, f: impl Fn(usize) -> f32) -> Tensor {
    let mut w = vec![0.0; 5120 * n];
    for j in 0..n {
        w[j] = f(j);
    }
    upload(ctx, vec![5120, n], w)
}
#[test]
fn controlled_gdn_decode_matches_output_and_f32_states() {
    let ctx = MusaContext::new(0).unwrap();
    let mut out = vec![0.0; 6144 * 5120];
    for j in 0..5120 {
        out[j * 5120 + j] = 0.25;
    }
    let weights = Qwen38GdnBf16Weights {
        input_norm: constant(&ctx, vec![5120], 0.5),
        qkv: projection(&ctx, 10240, |_| 0.5),
        z: projection(&ctx, 6144, |j| [-0.5, 0.0, 0.5][j % 3]),
        a: projection(&ctx, 48, |_| 0.25),
        b: projection(&ctx, 48, |_| -0.25),
        conv: constant(&ctx, vec![10240, 4], 0.25),
        a_log: constant(&ctx, vec![48], 0.0),
        dt_bias: constant(&ctx, vec![48], 0.125),
        gdn_norm: constant(&ctx, vec![128], 0.75),
        out: upload(&ctx, vec![6144, 5120], out),
    };
    let block = Qwen38GdnBf16Block::new(weights);
    let mut state = Qwen38GdnBf16State::zeroed(&ctx).unwrap();
    let input = constant(&ctx, vec![1, 5120], 1.0);
    let mut s = 0.0f32;
    for token in 1..=2 {
        let result = block.forward_decode(&ctx, &input, &mut state).unwrap();
        let values = transfers::to_cpu(&result).unwrap();
        assert_eq!(result.dtype(), DType::BF16);
        let normalized = round(1.5 / (1.0f32 + 1e-6).sqrt());
        let qkv = round(normalized * 0.5);
        let conv_x = qkv * 0.25 * token as f32;
        let v = round(conv_x / (1.0 + (-conv_x).exp()));
        let q = round(v / (128.0 * v * v + 1e-6).sqrt());
        let a = round(normalized * 0.25);
        let b = -a;
        let decay = (-(1.0 + (a + 0.125).exp()).ln()).exp();
        let beta = 1.0 / (1.0 + (-b).exp());
        s *= decay;
        s += (v - 128.0 * s * q) * beta * q;
        let readout = round(128.0 * s * q / 128.0f32.sqrt());
        let norm = round(round(readout / (readout * readout + 1e-6).sqrt()) * 0.75);
        for (j, actual) in values.as_bf16().unwrap().iter().enumerate() {
            let z = round(normalized * [-0.5, 0.0, 0.5][j % 3]);
            let silu = z / (1.0 + (-z).exp());
            let expected = round(1.0 + round(round(norm * silu) * 0.25));
            assert!(
                (actual.to_f32() - expected).abs() < 0.009,
                "token {token}, column {j}: {} vs {expected}",
                actual.to_f32()
            );
        }
        let recurrent = transfers::to_cpu(&state.recurrent).unwrap();
        for &actual in recurrent.as_f32().unwrap() {
            assert!((actual - s).abs() < 2e-5, "state {actual} vs {s}");
        }
        let window = transfers::to_cpu(&state.conv_window).unwrap();
        for row in window.as_f32().unwrap().chunks_exact(4) {
            for (i, &actual) in row.iter().enumerate() {
                assert_eq!(actual, if i >= 4 - token { qkv } else { 0.0 });
            }
        }
    }
}

#[test]
fn hf_gdn_loader_reports_missing_tensor_by_name() {
    let ctx = MusaContext::new(0).unwrap();
    let error = match Qwen38GdnBf16Weights::from_hf_layer(
        &ctx, 0, std::collections::HashMap::new(),
    ) {
        Ok(_) => panic!("empty checkpoint must not load"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("input_layernorm.weight"), "{error}");
}

#[test]
#[ignore = "requires official HF checkpoint and a MUSA device"]
fn real_hf_full_layer_zero_embedding_tokens_and_reset_are_finite_and_repeatable() {
    use std::collections::{HashMap, BTreeSet};
    let root=std::path::PathBuf::from(std::env::var("APXINF_QWEN38_CHECKPOINT").expect("checkpoint path"));
    let index:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json")).unwrap()).unwrap();
    let prefix="model.language_model.layers.0.";
    let embed_key="model.language_model.embed_tokens.weight";
    let wanted=|name:&str|name.starts_with(prefix) || name==embed_key;
    let wm=index["weight_map"].as_object().unwrap();
    let shards:BTreeSet<String>=wm.iter().filter(|(n,_)|wanted(n)).map(|(_,v)|v.as_str().unwrap().to_owned()).collect();
    let mut tensors=HashMap::new();
    for shard in shards {
        let (loaded,_)=apxinf_loader::safetensors::load_native(&root.join(shard)).unwrap();
        tensors.extend(loaded.into_iter().filter(|(name,_)|wanted(name)));
    }
    let embedding=tensors.remove(embed_key).expect("HF embedding table");
    assert_eq!(embedding.shape().dims(), &[248320,5120]);
    let token_ids=[1usize,42usize];
    let inputs:Vec<Tensor>=token_ids.iter().map(|&id| Tensor::from_bf16(vec![1,5120], &embedding.as_bf16().unwrap()[id*5120..(id+1)*5120]).unwrap()).collect();
    drop(embedding);
    assert_eq!(tensors.len(),14,"full GDN layer tensor inventory for {prefix}");
    let ctx=MusaContext::new(0).unwrap();
    let (mlp_tensors,gdn_tensors)=tensors.into_iter().partition(|(name,_)| name.contains(".mlp.") || name.ends_with("post_attention_layernorm.weight"));
    let mlp=apxinf_model::musa_qwen38::model::mlp::Qwen38MlpBf16::from_hf_layer(&ctx,0,mlp_tensors).unwrap();
    let weights=Qwen38GdnBf16Weights::from_hf_layer(&ctx,0,gdn_tensors).unwrap();
    let block=Qwen38GdnBf16Block::new(weights);
    let mut state=Qwen38GdnBf16State::zeroed(&ctx).unwrap();
    let mut first=None;
    for run in 0..2 {
        if run==1 {state=Qwen38GdnBf16State::zeroed(&ctx).unwrap();}
        let mut outputs=Vec::new();
        for token in 0..2 {
            let input=transfers::to_device(&inputs[token],ctx.device_id()).unwrap();
            let mixed=block.forward_decode(&ctx,&input,&mut state).unwrap();
            let output=mlp.forward(&ctx,&mixed).unwrap();
            ctx.synchronize().unwrap();
            let host=transfers::to_cpu(&output).unwrap();
            let values:Vec<_>=host.as_bf16().unwrap().iter().map(|x|x.to_f32()).collect();
            assert!(values.iter().all(|v|v.is_finite()));
            assert!(values.iter().any(|v|v.abs()>0.));
            println!("real layer0 HF embedding run={run} token_id={} output_abs_max={}",token_ids[token],values.iter().map(|v|v.abs()).fold(0f32,f32::max));
            outputs.push(values);
        }
        let host=transfers::to_cpu(&state.recurrent).unwrap();
        assert!(host.as_f32().unwrap().iter().all(|v|v.is_finite()));
        if let Some(ref previous)=first {assert_eq!(&outputs,previous,"reset must reproduce output");} else {first=Some(outputs);}
    }
}

#[test]
fn asymmetric_gdn_heads_match_independent_recurrence() {
    let ctx=MusaContext::new(0).unwrap();
    let pattern=|i:usize| -> f32 {
        if i<2048 { ((i*7%19) as f32-9.)/32. }
        else if i<4096 { ((i*11%23) as f32-11.)/32. }
        else { ((i*13%29) as f32-14.)/32. }
    };
    let taps=[-0.125,0.25,-0.5,0.75];
    let weights=Qwen38GdnBf16Weights {
        input_norm:constant(&ctx,vec![5120],0.),
        qkv:projection(&ctx,10240,pattern), z:projection(&ctx,6144,|i|(i%7) as f32/8.-0.5),
        a:projection(&ctx,48,|h|h as f32/128.-0.2),
        b:projection(&ctx,48,|h|0.3-h as f32/128.),
        conv:upload(&ctx,vec![10240,4],(0..40960).map(|i|taps[i%4]).collect()),
        a_log:constant(&ctx,vec![48],0.), dt_bias:constant(&ctx,vec![48],0.125),
        gdn_norm:constant(&ctx,vec![128],1.),out:constant(&ctx,vec![6144,5120],0.),
    };
    let block=Qwen38GdnBf16Block::new(weights);
    let mut state=Qwen38GdnBf16State::zeroed(&ctx).unwrap();
    let mut expected=vec![0f64;48*128*128];
    let mut history=vec![[0f32;4];10240];
    for token in 0..3 {
        let sign=if token==1 {-1.}else{1.};
        let input=constant(&ctx,vec![1,5120],sign);
        block.forward_decode(&ctx,&input,&mut state).unwrap();ctx.synchronize().unwrap();
        let normalized=round(sign/(1f32+1e-6).sqrt());
        let mut conv=vec![0f32;10240];
        for c in 0..10240 {
            history[c].rotate_left(1);history[c][3]=round(normalized*round(pattern(c)));
            let x=(0..4).map(|i|history[c][i]*taps[i]).sum::<f32>();
            conv[c]=round(x/(1.+(-x).exp()));
        }
        let mut k=conv[2048..4096].to_vec();
        for row in k.chunks_mut(128) {let norm=(row.iter().map(|x|x*x).sum::<f32>()+1e-6).sqrt();for x in row {*x=round(*x/norm);}}
        for h in 0..48 {
            let a=round(normalized*round(h as f32/128.-0.2));
            let b=round(normalized*round(0.3-h as f32/128.));
            let decay=(-(1.+(a as f64+0.125).exp()).ln()).exp();
            let beta=1./(1.+(-b as f64).exp());
            for v in 0..128 {
                let base=(h*128+v)*128;
                let keys=&k[(h/3)*128..(h/3+1)*128];
                let mut pred=0.;for d in 0..128 {expected[base+d]*=decay;pred+=expected[base+d]*keys[d] as f64;}
                let delta=(conv[4096+h*128+v] as f64-pred)*beta;
                for d in 0..128 {expected[base+d]+=delta*keys[d] as f64;}
            }
        }
        let actual=transfers::to_cpu(&state.recurrent).unwrap();
        for (i,(&a,&e)) in actual.as_f32().unwrap().iter().zip(&expected).enumerate() {
            assert!((a as f64-e).abs()<2e-5,"token {token} state {i}: {a} vs {e}");
        }
        let window=transfers::to_cpu(&state.conv_window).unwrap();
        for (c,row) in window.as_f32().unwrap().chunks(4).enumerate(){assert_eq!(row,&history[c]);}
    }
}
