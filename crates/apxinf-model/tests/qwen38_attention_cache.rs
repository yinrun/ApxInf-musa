#![cfg(feature="musa")]
use apxinf_core::Tensor;
use apxinf_musa::{MusaContext,transfers};
use apxinf_model::musa_qwen38::model::attention::KvState;
#[test]
fn cache_append_capacity_and_reset_preserve_valid_prefix() {
 let ctx=MusaContext::new(0).unwrap();
 let mut cache=KvState::new(&ctx,2).unwrap();
 for step in 0..2 {
  let host=Tensor::from_bf16(vec![1,4,256],&vec![half::bf16::from_f32((step+1) as f32);1024]).unwrap();
  let token=transfers::to_device(&host,0).unwrap();
  cache.append(&ctx,&token,&token).unwrap();
  let (k,_)=cache.prefix().unwrap();
  assert_eq!(k.shape().dims(), &[1,step+1,4,256]);
  let got=transfers::to_cpu(&k).unwrap();
  for row in 0..=step {assert!(got.as_bf16().unwrap()[row*1024..(row+1)*1024].iter().all(|x|x.to_f32()==(row+1) as f32));}
 }
 let host=Tensor::from_bf16(vec![1,4,256],&vec![half::bf16::from_f32(9.);1024]).unwrap();
 let token=transfers::to_device(&host,0).unwrap();
 assert!(cache.append(&ctx,&token,&token).is_err());
 cache.reset();assert!(cache.prefix().is_err());
 cache.append(&ctx,&token,&token).unwrap();
 let(k,_)=cache.prefix().unwrap();assert_eq!(k.shape().dims(),&[1,1,4,256]);
 assert!(transfers::to_cpu(&k).unwrap().as_bf16().unwrap().iter().all(|x|x.to_f32()==9.));
}

#[test]
fn attention_decode_uniform_scores_match_two_token_reference() {
 use apxinf_model::musa_qwen38::model::attention::AttentionBlock;
 use half::bf16;
 let ctx=MusaContext::new(0).unwrap();
 let upload=|dims:Vec<usize>,x:Vec<f32>| transfers::to_device(&Tensor::from_bf16(dims,&x.into_iter().map(bf16::from_f32).collect::<Vec<_>>()).unwrap(),0).unwrap();
 let zeros=|dims:Vec<usize>| {let n=dims.iter().product();upload(dims,vec![0.;n])};
 let mut v=vec![0.;5120*1024];for j in 0..1024 {v[j]=((j/256)+1) as f32/8.;}
 let mut o=vec![0.;6144*5120];for j in 0..5120 {o[j*5120+j]=0.5;}
 let block=AttentionBlock {input_norm:zeros(vec![5120]),q:zeros(vec![5120,12288]),k:zeros(vec![5120,1024]),v:upload(vec![5120,1024],v),o:upload(vec![6144,5120],o),q_norm:zeros(vec![256]),k_norm:zeros(vec![256])};
 let mut cache=KvState::new(&ctx,2).unwrap();
 let round=|x:f32|bf16::from_f32(x).to_f32();
 for token in 0..2 {
  let x=if token==0 {1.}else{-1.};
  let input=upload(vec![1,5120],vec![x;5120]);
  let output=block.forward_decode(&ctx,&input,&mut cache).unwrap();ctx.synchronize().unwrap();
  let host=transfers::to_cpu(&output).unwrap();
  for (j,got) in host.as_bf16().unwrap().iter().enumerate() {
   let value=if token==0 {round(round(1./(1f32+1e-6).sqrt())*((j/256/6)+1) as f32/8.)}else{0.};
   let want=round(x+round(round(value*0.5)*0.5));
   assert!((got.to_f32()-want).abs()<0.009,"token {token} col {j}: {} vs {want}",got.to_f32());
  }
  assert_eq!(cache.len(),token+1);
 }
}

#[test]
fn attention_hf_loader_refuses_missing_norm() {
 use apxinf_model::musa_qwen38::model::attention::AttentionBlock;
 let ctx=MusaContext::new(0).unwrap();
 match AttentionBlock::from_hf_layer(&ctx,3,std::collections::HashMap::new()) {
  Ok(_)=>panic!("empty layer accepted"),
  Err(e)=>assert!(e.to_string().contains("input_layernorm.weight"),"{e}"),
 }
}

#[test]
#[ignore = "requires official HF checkpoint and S5000"]
fn real_hf_attention_layer_three_is_finite_and_reset_repeatable() {
 use apxinf_model::musa_qwen38::model::{attention::AttentionBlock,mlp::Qwen38MlpBf16};
 use std::collections::{HashMap,BTreeSet};
 let root=std::path::PathBuf::from(std::env::var("APXINF_QWEN38_CHECKPOINT").unwrap());
 let index:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json")).unwrap()).unwrap();
 let prefix="model.language_model.layers.3.";
 let shards:BTreeSet<_>=index["weight_map"].as_object().unwrap().iter().filter(|(n,_)|n.starts_with(prefix)).map(|(_,v)|v.as_str().unwrap().to_owned()).collect();
 let mut tensors=HashMap::new();
 for shard in shards {let (map,_)=apxinf_loader::safetensors::load_native(&root.join(shard)).unwrap();tensors.extend(map.into_iter().filter(|(n,_)|n.starts_with(prefix)));}
 assert_eq!(tensors.len(),11);
 let (mlp,attn)=tensors.into_iter().partition(|(n,_)|n.contains(".mlp.")||n.ends_with("post_attention_layernorm.weight"));
 let ctx=MusaContext::new(0).unwrap();
 let mlp=Qwen38MlpBf16::from_hf_layer(&ctx,3,mlp).unwrap();
 let attn=AttentionBlock::from_hf_layer(&ctx,3,attn).unwrap();
 let mut cache=KvState::new(&ctx,2).unwrap();let mut previous=None;
 for run in 0..2 {
  cache.reset();let mut outputs=Vec::new();
  for token in 0..2 {
   let values:Vec<_>=(0..5120).map(|i|half::bf16::from_f32(((i*7+token*11)%53) as f32/53.-0.5)).collect();
   let input=transfers::to_device(&Tensor::from_bf16(vec![1,5120],&values).unwrap(),0).unwrap();
   let mixed=attn.forward_decode(&ctx,&input,&mut cache).unwrap();
   let output=mlp.forward(&ctx,&mixed).unwrap();ctx.synchronize().unwrap();
   let host=transfers::to_cpu(&output).unwrap();let values:Vec<_>=host.as_bf16().unwrap().iter().map(|x|x.to_f32()).collect();
   assert!(values.iter().all(|x|x.is_finite()));
   println!("layer3 run={run} token={token} output_abs_max={}",values.iter().map(|x|x.abs()).fold(0f32,f32::max));outputs.push(values);
  }
  if let Some(ref p)=previous {assert_eq!(&outputs,p);}else{previous=Some(outputs);}
 }
}

#[test]
fn nonzero_attention_matches_positioned_cpu_reference() {
 use apxinf_model::musa_qwen38::model::attention::AttentionBlock;
 use half::bf16;
 let ctx=MusaContext::new(0).unwrap();
 let round=|x:f32|bf16::from_f32(x).to_f32();
 let upload=|dims:Vec<usize>,v:Vec<f32>|transfers::to_device(&Tensor::from_bf16(dims,&v.into_iter().map(bf16::from_f32).collect::<Vec<_>>()).unwrap(),0).unwrap();
 let projection=|n:usize,f:&dyn Fn(usize)->f32|{let mut v=vec![0.;5120*n];for j in 0..n {v[j]=f(j);}upload(vec![5120,n],v)};
 let qp=|i:usize|((i*7%23) as f32-11.)/16.;
 let kp=|i:usize|((i*11%19) as f32-9.)/16.;
 let vp=|i:usize|((i*3%17) as f32-8.)/16.;
 let norm:Vec<_>=(0..256).map(|d|round((d%7) as f32*0.003)).collect();
 let mut o=vec![0.;6144*5120];for j in 0..5120 {o[j*5120+j]=0.25;}
 let block=AttentionBlock{input_norm:upload(vec![5120],vec![0.;5120]),q:projection(12288,&qp),k:projection(1024,&kp),v:projection(1024,&vp),o:upload(vec![6144,5120],o),q_norm:upload(vec![256],norm.clone()),k_norm:upload(vec![256],norm.clone())};
 let normalize_rotate=|mut x:Vec<f32>,heads:usize,pos:usize| {
  for h in 0..heads {
   let row=&mut x[h*256..(h+1)*256];
   let mean=row.iter().map(|&v|(v as f64).powi(2)).sum::<f64>()/256.;
   for d in 0..256 {row[d]=round((row[d] as f64/(mean+1e-6).sqrt()*(1.+norm[d] as f64)) as f32);}
   for d in 0..32 {let angle=pos as f64*1e7f64.powf(-2.*d as f64/64.);let(a,b)=(row[d] as f64,row[d+32] as f64);row[d]=round((a*angle.cos()-b*angle.sin()) as f32);row[d+32]=round((b*angle.cos()+a*angle.sin()) as f32);}
  } x
 };
 for _run in 0..2 {
  let mut cache=KvState::new(&ctx,3).unwrap();let mut keys=Vec::new();let mut vals=Vec::new();
  for t in 0..3 {
   let sign=if t==1 {-1.}else{1.};let n=round(sign/(1f32+1e-6).sqrt());
   let fused:Vec<_>=(0..12288).map(|i|round(n*round(qp(i)))).collect();
   let mut query=Vec::new();let mut gate=Vec::new();for h in 0..24 {query.extend_from_slice(&fused[h*512..h*512+256]);gate.extend_from_slice(&fused[h*512+256..(h+1)*512]);}
   let query=normalize_rotate(query,24,t);
   keys.push(normalize_rotate((0..1024).map(|i|round(n*round(kp(i)))).collect(),4,t));
   vals.push((0..1024).map(|i|round(n*round(vp(i)))).collect::<Vec<_>>());
   let input=upload(vec![1,5120],vec![sign;5120]);
   let output=block.forward_decode(&ctx,&input,&mut cache).unwrap();ctx.synchronize().unwrap();
   let host=transfers::to_cpu(&output).unwrap();let got=host.as_bf16().unwrap();
   let(mut diff,mut energy)=(0f64,0f64);
   for h in 0..20 {
    let mut scores:Vec<_>=(0..=t).map(|j|(0..256).map(|d|query[h*256+d] as f64*keys[j][(h/6)*256+d] as f64).sum::<f64>()/16.).collect();
    let max=scores.iter().copied().fold(f64::NEG_INFINITY,f64::max);for s in &mut scores {*s=(*s-max).exp();}let sum=scores.iter().sum::<f64>();
    for d in 0..256 {let i=h*256+d;let v=(0..=t).map(|j|scores[j]*vals[j][(h/6)*256+d] as f64).sum::<f64>()/sum;
     let gated=round(round(v as f32)/(1.+(-gate[i]).exp()));let want=round(sign+round(gated*0.25));let actual=got[i].to_f32();assert!(actual.is_finite());diff+=(actual as f64-want as f64).powi(2);energy+=(want as f64).powi(2);
    }
   }
   let error=(diff/energy).sqrt();assert!(error<0.005,"position {t} relative L2 {error}");
  }
 }
}
