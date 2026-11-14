//! G128 symmetric signed INT4 weights and BF16 activations, FP32 accumulation.
use apxinf_core::{DType, Error, Result, Tensor};
use crate::{MusaBuffer, MusaContext};

pub struct W4A16WeightView<'a> {
    pub packed: &'a MusaBuffer,
    pub scales: &'a Tensor,
    pub k: usize,
    pub n: usize,
    pub group_size: usize,
}

pub fn linear(ctx: &MusaContext, a: &Tensor, w: W4A16WeightView<'_>, out: &mut Tensor) -> Result<()> {
    let dims = a.shape().dims();
    let group = w.group_size;
    if group == 0 { return Err(Error::Other("W4 group size must be non-zero".into())); }
    let kp = w.k.checked_add(group - 1).ok_or_else(|| Error::Other("W4 K overflow".into()))? / group * group;
    if a.device()!=apxinf_core::Device::Musa(ctx.device_id()) || out.device()!=a.device() || w.scales.device()!=a.device() {
        return Err(Error::Other("W4 tensors must belong to the context device".into()));
    }
    if dims.len()!=2 || dims[1]!=w.k || a.dtype()!=DType::BF16 || out.dtype()!=DType::BF16
        || out.shape().dims()!=[dims[0],w.n] || w.scales.dtype()!=DType::BF16
        || w.scales.shape().dims()!=[w.n,kp/group] {
        return Err(Error::Other("W4A16 shape/dtype mismatch".into()));
    }
    if w.packed.len() < w.n.checked_mul(kp/2).ok_or_else(|| Error::Other("W4 byte size overflow".into()))? || w.packed.device()!=ctx.device_id() {
        return Err(Error::Other("W4 packed storage size/device mismatch".into()));
    }
    #[cfg(apxinf_musa_available)]
    {
        let aa=MusaBuffer::from_tensor(a).map_err(Error::Other)?;
        let ss=MusaBuffer::from_tensor(w.scales).map_err(Error::Other)?;
        let cc=MusaBuffer::from_tensor(out).map_err(Error::Other)?;
        let native_pair_shape = (dims[0] == 522 && matches!((w.n,w.k),(32768,2048)|(2048,16384)))
            || (dims[0] == 512 && matches!((w.n,w.k),
                (3456,1152)|(1152,1152)|(4304,1152)|(1152,4304)));
        if group==16 && native_pair_shape && crate::workspace::w4_native_enabled() {
            let scratch=crate::workspace::w4_session_buffer(ctx,w.n,w.k)?;
            crate::workspace::note_shape(crate::workspace::OperationShape::Gemm {m:dims[0],n:w.n,k:w.k})?;
            unsafe {
                crate::ffi::check_runtime(crate::ffi::custom::apxinf_musa_w4_decode_kn(w.packed.ptr(),ss.ptr(),scratch.ptr(),w.n as i32,w.k as i32,ctx.stream().handle())).map_err(Error::Other)?;
                crate::ffi::mublas::check_mublas(crate::ffi::mublas::apxinf_musa_bf16_gemm(aa.ptr(),scratch.ptr(),cc.ptr(),dims[0] as i32,w.n as i32,w.k as i32,1.,ctx.stream().handle())).map_err(Error::Other)?;
            }
            return Ok(());
        }
        crate::kernels::gemm::launch_w4(ctx,dims[0],w.n,w.k,group,aa.ptr(),w.packed.ptr(),ss.ptr(),cc.ptr())
    }
    #[cfg(not(apxinf_musa_available))]
    { let _=(ctx,w,kp); Err(Error::Other("W4A16 requires MUSA TileLang".into())) }
}
