#![cfg(feature = "musa")]
use apxinf_core::Tensor;
use apxinf_model::musa_qwen38::model::mlp::Qwen38MlpBf16;
use apxinf_musa::{transfers,MusaContext};
use half::bf16;
fn upload(ctx:&MusaContext,dims:Vec<usize>,x:Vec<f32>)->Tensor {
    transfers::to_device(&Tensor::from_bf16(dims,&x.into_iter().map(bf16::from_f32).collect::<Vec<_>>()).unwrap(),ctx.device_id()).unwrap()
}
fn round(x:f32)->f32{bf16::from_f32(x).to_f32()}
#[test]
fn mlp_matches_sparse_nonzero_reference() {
    let ctx=MusaContext::new(0).unwrap();
    let mut gate=vec![0.;5120*17408];let mut up=gate.clone();
    for j in 0..17408 {gate[j]=[-0.5,0.,0.5][j%3];up[j]=0.25;}
    let mut down=vec![0.;17408*5120];for j in 0..5120 {down[j*5120+j]=0.5;}
    let block=Qwen38MlpBf16 {
        norm:upload(&ctx,vec![5120],vec![0.5;5120]),
        gate:upload(&ctx,vec![5120,17408],gate),
        up:upload(&ctx,vec![5120,17408],up),
        down:upload(&ctx,vec![17408,5120],down),
    };
    let input=upload(&ctx,vec![1,5120],vec![1.;5120]);
    let output=block.forward(&ctx,&input).unwrap();ctx.synchronize().unwrap();
    let host=transfers::to_cpu(&output).unwrap();
    let normalized=round(1.5/(1f32+1e-6).sqrt());
    for (j,x) in host.as_bf16().unwrap().iter().enumerate() {
        let g=round(normalized*[-0.5,0.,0.5][j%3]);let u=round(normalized*0.25);
        let want=round(1.+round(round(g/(1.+(-g).exp())*u)*0.5));
        assert!((x.to_f32()-want).abs()<0.009,"column {j}: {} vs {want}",x.to_f32());
    }
}

#[test]
fn mlp_hf_loader_rejects_missing_norm() {
    let ctx=MusaContext::new(0).unwrap();
    let result=Qwen38MlpBf16::from_hf_layer(&ctx,0,std::collections::HashMap::new());
    match result { Ok(_)=>panic!("empty map accepted"), Err(e)=>assert!(e.to_string().contains("post_attention_layernorm.weight"),"{e}") }
}
