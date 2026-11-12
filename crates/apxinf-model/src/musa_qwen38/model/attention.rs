//! Fixed-capacity, single-sequence KV storage for eager Qwen attention.
use apxinf_core::{DType,Device,Error,Result,Shape,Tensor};
use apxinf_musa::{MusaBuffer,MusaContext};

pub struct KvState {
    keys: MusaBuffer,
    values: MusaBuffer,
    capacity: usize,
    valid: usize,
    poisoned: bool,
    device: usize,
}
impl KvState {
    pub fn new(ctx:&MusaContext,capacity:usize)->Result<Self> {
        let bytes=capacity.checked_mul(4*256*2).filter(|&b|b>0)
            .ok_or_else(||Error::Other("KV capacity must be positive and fit address space".into()))?;
        Ok(Self {keys:MusaBuffer::alloc(bytes,ctx.device_id()).map_err(Error::Other)?,
            values:MusaBuffer::alloc(bytes,ctx.device_id()).map_err(Error::Other)?,
            capacity,valid:0,poisoned:false,device:ctx.device_id()})
    }
    pub fn len(&self)->usize {self.valid}
    pub fn is_empty(&self)->bool {self.valid==0}
    /// Old bytes remain allocated but are never exposed until overwritten.
    pub fn reset(&mut self){self.valid=0;self.poisoned=false;}
    pub fn append(&mut self,ctx:&MusaContext,key:&Tensor,value:&Tensor)->Result<()> {
        if self.poisoned || ctx.device_id()!=self.device || self.valid>=self.capacity {
            return Err(Error::Other("KV context mismatch or capacity exhausted".into()));
        }
        for t in [key,value] {
            if t.dtype()!=DType::BF16 || t.shape().dims()!=[1,4,256] || t.device()!=Device::Musa(self.device) {
                return Err(Error::Other("KV append expects BF16 [1,4,256] on cache device".into()));
            }
        }
        self.poisoned=true;
        let bytes=4*256*2;
        for (destination,source) in [(&self.keys,key),(&self.values,value)] {
            let source=MusaBuffer::from_tensor(source).map_err(Error::Other)?;
            destination.view(self.valid*bytes,bytes).map_err(Error::Other)?
                .copy_from_device_async(&source,bytes,ctx.stream()).map_err(Error::Other)?;
        }
        self.valid+=1;
        self.poisoned=false;
        Ok(())
    }
    // Execute the actual post-append region under a persistent failure marker.
    fn with_appended<T>(&mut self,ctx:&MusaContext,key:&Tensor,value:&Tensor,
        operation:impl FnOnce(&Self)->Result<T>)->Result<T> {
        self.append(ctx,key,value)?;
        let result=operation(self);
        if result.is_err(){self.poisoned=true;}
        result
    }
    pub fn prefix(&self)->Result<(Tensor,Tensor)> {
        if self.poisoned || self.valid==0 {return Err(Error::Other("KV cache is empty".into()));}
        let as_tensor=|buffer:&MusaBuffer| -> Result<Tensor> {
            buffer.view(0,self.valid*4*256*2).map_err(Error::Other)?
                .as_tensor(Shape::new(vec![1,self.valid,4,256]),DType::BF16).map_err(Error::Other)
        };
        Ok((as_tensor(&self.keys)?,as_tensor(&self.values)?))
    }
}

/// Eager single-token full-attention mixer plus residual (MLP is separate).
/// Linear weights use KN layout; normalization weights are raw HF offsets.
pub struct AttentionBlock {
    pub input_norm: Tensor,
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub o: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

fn offset_norm(ctx:&MusaContext,input:&Tensor,weight:&Tensor)->Result<Tensor> {
    let mut out=ctx.allocate_output(input.shape().clone(),DType::BF16)?;
    apxinf_musa::kernels::qwen38::offset_rms_bf16(ctx,input,weight,&mut out,1e-6)?;
    Ok(out)
}

impl AttentionBlock {
    /// Load the seven native HF BF16 attention tensors, transposing linear weights.
    pub fn from_hf_layer(ctx: &MusaContext, layer: usize, mut tensors: std::collections::HashMap<String,Tensor>) -> Result<Self> {
        if layer >= 64 || (layer + 1) % 4 != 0 { return Err(Error::Other("not a full-attention layer".into())); }
        let mut take = |suffix: &str, shape: &[usize], transpose: bool| -> Result<Tensor> {
            let key=format!("model.language_model.layers.{layer}.{suffix}");
            let tensor=tensors.remove(&key).ok_or_else(||Error::Other(format!("missing {key}")))?;
            if tensor.dtype()!=DType::BF16 || tensor.shape().dims()!=shape {
                return Err(Error::Other(format!("{key}: expected BF16 {shape:?}")));
            }
            let tensor=if transpose {
                let source=tensor.as_bf16()?;
                let (rows,cols)=(shape[0],shape[1]);
                let mut result=vec![half::bf16::ZERO;source.len()];
                for r in 0..rows {for c in 0..cols {result[c*rows+r]=source[r*cols+c];}}
                Tensor::from_bf16(vec![cols,rows],&result)?
            } else {tensor};
            apxinf_musa::transfers::to_device(&tensor,ctx.device_id())
        };
        Ok(Self {
            input_norm:take("input_layernorm.weight",&[5120],false)?,
            q:take("self_attn.q_proj.weight",&[12288,5120],true)?,
            k:take("self_attn.k_proj.weight",&[1024,5120],true)?,
            v:take("self_attn.v_proj.weight",&[1024,5120],true)?,
            o:take("self_attn.o_proj.weight",&[5120,6144],true)?,
            q_norm:take("self_attn.q_norm.weight",&[256],false)?,
            k_norm:take("self_attn.k_norm.weight",&[256],false)?,
        })
    }
    /// Append one rotated K/V token; a returned post-append error poisons the cache until reset.
    /// Asynchronous device faults require discarding the request state.
    pub fn forward_decode(&self,ctx:&MusaContext,input:&Tensor,cache:&mut KvState)->Result<Tensor> {
        use apxinf_musa::{kernels::qwen38,transfers};
        if cache.poisoned || cache.device!=ctx.device_id() || cache.valid>=cache.capacity {
            return Err(Error::Other("attention cache device mismatch or exhausted".into()));
        }
        for (name,t,shape) in [
            ("input",input,vec![1,5120]),("input_norm",&self.input_norm,vec![5120]),
            ("q",&self.q,vec![5120,12288]),("k",&self.k,vec![5120,1024]),
            ("v",&self.v,vec![5120,1024]),("o",&self.o,vec![6144,5120]),
            ("q_norm",&self.q_norm,vec![256]),("k_norm",&self.k_norm,vec![256]),
        ] {
            if t.dtype()!=DType::BF16 || t.shape().dims()!=shape || t.device()!=Device::Musa(ctx.device_id()) {
                return Err(Error::Other(format!("invalid attention {name}: expected BF16 {shape:?} on context device")));
            }
        }
        let position=i32::try_from(cache.valid).map_err(|_|Error::Other("position exceeds i32".into()))?;
        let normalized=offset_norm(ctx,input,&self.input_norm)?;
        let fused=super::gemm(ctx,&normalized,&self.q,vec![1,12288])?.reshape(vec![24,512])?;
        let mut query=ctx.allocate_output(Shape::new(vec![24,256]),DType::BF16)?;
        let mut gate=ctx.allocate_output(Shape::new(vec![24,256]),DType::BF16)?;
        qwen38::split_query_and_gate(ctx,&fused,&mut query,&mut gate)?;
        let key=super::gemm(ctx,&normalized,&self.k,vec![1,1024])?.reshape(vec![4,256])?;
        let value=super::gemm(ctx,&normalized,&self.v,vec![1,1024])?.reshape(vec![1,4,256])?;
        let mut query=offset_norm(ctx,&query,&self.q_norm)?.reshape(vec![1,24,256])?;
        let mut key=offset_norm(ctx,&key,&self.k_norm)?.reshape(vec![1,4,256])?;
        let positions=Tensor::from_raw(Shape::new(vec![1,3]),DType::I32,Device::Cpu,
            [position;3].iter().flat_map(|x|x.to_le_bytes()).collect())?;
        let positions=transfers::to_device(&positions,ctx.device_id())?;
        qwen38::rope(ctx,&mut query,&positions,64,[11,11,10],1e7)?;
        qwen38::rope(ctx,&mut key,&positions,64,[11,11,10],1e7)?;
        cache.with_appended(ctx,&key,&value,|cache| {
        let (keys,values)=cache.prefix()?;
        let mut attended=ctx.allocate_output(Shape::new(vec![1,1,24,256]),DType::BF16)?;
        qwen38::sdpa_bf16(ctx,&query.reshape(vec![1,1,24,256])?,&keys,&values,&mut attended,false)?;
        let mut attended=attended.reshape(vec![24,256])?;
        qwen38::apply_output_gate(ctx,&mut attended,&gate)?;
        let mut out=super::gemm(ctx,&attended.reshape(vec![1,6144])?,&self.o,vec![1,5120])?;
        qwen38::add_into(ctx,&mut out,input)?;
        Ok(out)
        })
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    #[test]
    fn post_append_failure_poison_requires_reset() {
        let ctx=MusaContext::new(0).unwrap();
        let host=Tensor::from_bf16(vec![1,4,256],&vec![half::bf16::ZERO;1024]).unwrap();
        let token=apxinf_musa::transfers::to_device(&host,0).unwrap();
        let mut cache=KvState::new(&ctx,2).unwrap();
        let failed:Result<()>=cache.with_appended(&ctx,&token,&token,|_|Err(Error::Other("injected downstream failure".into())));
        assert!(failed.is_err());
        assert!(cache.prefix().is_err());
        assert!(cache.append(&ctx,&token,&token).is_err());
        cache.reset();
        cache.with_appended(&ctx,&token,&token,|c|{assert_eq!(c.len(),1);Ok(())}).unwrap();
        assert!(cache.prefix().is_ok());
    }
}
