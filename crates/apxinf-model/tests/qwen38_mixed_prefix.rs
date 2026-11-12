#![cfg(feature="musa")]
use apxinf_core::Tensor;
use apxinf_musa::{MusaContext,transfers};
use apxinf_model::musa_qwen38::model::{Qwen38GdnBf16Block as Gdn,Qwen38GdnBf16Weights as GW,Qwen38GdnBf16State as GS,attention::{AttentionBlock,KvState},mlp::Qwen38MlpBf16 as Mlp};
use std::collections::{HashMap,BTreeSet};
#[test]
#[ignore="requires official HF checkpoint and S5000"]
fn first_four_hf_layers_preserve_finite_state_and_reset_reproducibility() {
 let root=std::path::PathBuf::from(std::env::var("APXINF_QWEN38_CHECKPOINT").unwrap());
 let index:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json")).unwrap()).unwrap();
 let embed="model.language_model.embed_tokens.weight";
 let wanted=|n:&str|n==embed || (0..4).any(|l|n.starts_with(&format!("model.language_model.layers.{l}.")));
 let shards:BTreeSet<_>=index["weight_map"].as_object().unwrap().iter().filter(|(n,_)|wanted(n)).map(|(_,v)|v.as_str().unwrap().to_owned()).collect();
 let mut tensors=HashMap::new();
 for shard in shards {let (map,_)=apxinf_loader::safetensors::load_native(&root.join(shard)).unwrap();tensors.extend(map.into_iter().filter(|(n,_)|wanted(n)));}
 let embedding=tensors.remove(embed).unwrap();
 let inputs:Vec<_>=[1usize,42,314].iter().map(|&id|Tensor::from_bf16(vec![1,5120],&embedding.as_bf16().unwrap()[id*5120..(id+1)*5120]).unwrap()).collect();drop(embedding);
 let ctx=MusaContext::new(0).unwrap();let mut gdns=Vec::new();let mut mlps=Vec::new();let mut attention=None;
 for l in 0..4 {
  let prefix=format!("model.language_model.layers.{l}.");
  let names:Vec<_>=tensors.keys().filter(|n|n.starts_with(&prefix)).cloned().collect();
  assert_eq!(names.len(),if l==3 {11}else{14});
  let layer:HashMap<_,_>=names.into_iter().map(|n|{let t=tensors.remove(&n).unwrap();(n,t)}).collect();
  let (mlp,mixer)=layer.into_iter().partition(|(n,_)|n.contains(".mlp.")||n.ends_with("post_attention_layernorm.weight"));
  mlps.push(Mlp::from_hf_layer(&ctx,l,mlp).unwrap());
  if l==3 {attention=Some(AttentionBlock::from_hf_layer(&ctx,l,mixer).unwrap());}else{gdns.push(Gdn::new(GW::from_hf_layer(&ctx,l,mixer).unwrap()));}
 }
 let attention=attention.unwrap();let mut reference=None;
 for run in 0..2 {
  let mut states:Vec<_>=(0..3).map(|_|GS::zeroed(&ctx).unwrap()).collect();let mut kv=KvState::new(&ctx,3).unwrap();let mut outputs=Vec::new();
  for (token,input) in inputs.iter().enumerate() {
   let mut hidden=transfers::to_device(input,0).unwrap();
   for l in 0..4 {
    hidden=if l<3 {gdns[l].forward_decode(&ctx,&hidden,&mut states[l]).unwrap()}else{attention.forward_decode(&ctx,&hidden,&mut kv).unwrap()};
    hidden=mlps[l].forward(&ctx,&hidden).unwrap();ctx.synchronize().unwrap();
    let host=transfers::to_cpu(&hidden).unwrap();let values:Vec<_>=host.as_bf16().unwrap().iter().map(|x|x.to_f32()).collect();
    assert!(values.iter().all(|x|x.is_finite()),"run={run} token={token} layer={l}");
    println!("run={run} token={token} layer={l} output_abs_max={}",values.iter().map(|x|x.abs()).fold(0f32,f32::max));outputs.push(values);
   }
   assert_eq!(kv.len(),token+1);
  }
  for s in states {assert!(transfers::to_cpu(&s.recurrent).unwrap().as_f32().unwrap().iter().all(|x|x.is_finite()));}
  if let Some(ref r)=reference {assert_eq!(&outputs,r,"fresh state must reproduce every layer");}else{reference=Some(outputs);}
 }
}
