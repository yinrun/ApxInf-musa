"""Fused NK packed signed INT4 G16 -> BF16 tiled GEMM, FP32 accumulation."""
# Explicit initial profile: two 256-patch views, ten language tokens.
OPS = [dict(id=f"pi05-w4-{m}-{n}-{k}", type="gemm-w4a16",
            supported_arches=["mp_22"], shape=(m,n,k),
            flags=["-Od3", "-O2"], status="enabled")
       for m,n,k in [(512,3456,1152),(512,1152,1152),
                     (512,4304,1152),(512,1152,4304),
                     (522,2560,2048),(522,2048,2048),
                     (522,32768,2048),(522,2048,16384)]]

def build_kernel(m, n, k, bm=None, bn=64, bk=64, num_stages=1, paired=None, threads=None):
    # Measured exact-shape policy; explicit tuning arguments bypass selection.
    selected = bm is None
    if bm is None:
        bm = 64 if (m, n, k) in {tuple(op["shape"]) for op in OPS} else 32
        if paired is None:
            paired = (m, n, k) in {(512, 4304, 1152), (512, 1152, 4304)}
    if paired is None:
        paired = False
    if selected and (m,n,k) in {(522,32768,2048),(522,2048,16384)}:
        bm = 128
        if threads is None:
            threads = 256
    if selected and (m,n,k) in {(512,3456,1152),(512,1152,1152),(512,4304,1152),(512,1152,4304)}:
        bm, threads, paired = 128, 256, True
    if selected and (m,n,k) in {(522,32768,2048),(522,2048,16384)}:
        bn, threads = 128, 512
    if threads is None:
        threads = 128
    reuse_group = selected and m == 512 and (m,n,k) in {tuple(op["shape"]) for op in OPS}
    import tilelang
    import tilelang.language as T
    group_size = 16
    kp = ((k + group_size - 1) // group_size) * group_size
    @tilelang.jit(out_idx=[], target="musa", compile_flags=["-Od3", "-O2"])
    def make():
        @T.prim_func
        def kernel(A: T.Tensor((m,k), "bfloat16"),
                   B: T.Tensor((n,kp//2), "uint8"),
                   S: T.Tensor((n,kp//group_size), "bfloat16"),
                   C: T.Tensor((m,n), "bfloat16")):
            with T.Kernel(T.ceildiv(n,bn),T.ceildiv(m,bm),threads=threads) as (bx,by):
                aa=T.alloc_shared((bm,bk),"bfloat16")
                bb=T.alloc_shared((bn,bk),"bfloat16")
                cc=T.alloc_fragment((bm,bn),"float32")
                T.clear(cc)
                for ki in T.Pipelined(T.ceildiv(k,bk),num_stages=num_stages):
                    T.copy(A[by*bm,ki*bk],aa)
                    if reuse_group:
                        for j,g in T.Parallel(bn,bk//16):
                            scale=T.Cast("float32",S[bx*bn+j,ki*bk//16+g])
                            for v in T.unroll(8):
                                byte=T.Cast("int32",B[bx*bn+j,ki*bk//2+g*8+v])
                                lo=byte & 15
                                hi=(byte >> 4) & 15
                                qlo=T.if_then_else(lo>=8,lo-16,lo)
                                qhi=T.if_then_else(hi>=8,hi-16,hi)
                                bb[j,g*16+2*v]=T.Cast("bfloat16",T.Cast("float32",qlo)*scale)
                                bb[j,g*16+2*v+1]=T.Cast("bfloat16",T.Cast("float32",qhi)*scale)
                    else:
                        if paired:
                            for j,t in T.Parallel(bn,bk//2):
                                byte=T.Cast("int32",B[bx*bn+j,ki*bk//2+t])
                                lo=byte & 15
                                hi=(byte >> 4) & 15
                                qlo=T.if_then_else(lo>=8,lo-16,lo)
                                qhi=T.if_then_else(hi>=8,hi-16,hi)
                                scale=T.Cast("float32",S[bx*bn+j,(ki*bk+2*t)//group_size])
                                bb[j,2*t]=T.Cast("bfloat16",T.Cast("float32",qlo)*scale)
                                bb[j,2*t+1]=T.Cast("bfloat16",T.Cast("float32",qhi)*scale)
                        else:
                            for j,t in T.Parallel(bn,bk):
                                byte=T.Cast("int32",B[bx*bn+j,(ki*bk+t)//2])
                                nib=(byte >> ((t%2)*4)) & 15
                                signed=T.if_then_else(nib>=8,nib-16,nib)
                                bb[j,t]=T.Cast("bfloat16",T.Cast("float32",signed)*T.Cast("float32",S[bx*bn+j,(ki*bk+t)//group_size]))
                    T.gemm(aa,bb,cc,transpose_B=True)
                T.copy(cc,C[by*bm,bx*bn])
        return kernel
    return make()
