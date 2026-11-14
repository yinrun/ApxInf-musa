"""Reference W4 G16 format. NK bytes, low nibble first; NK-group BF16 scales.

Round ties to even using the stored scale; pad K to 16 with zero nibbles.
All-zero groups use scale 1. Nonfinite inputs and nonnormal scales fail closed.
This CPU utility is a correctness oracle, not a device execution path.
"""
OPS = []

def quantize(weight):
    import numpy as np
    import ml_dtypes
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2 or not all(w.shape) or not np.isfinite(w).all():
        raise ValueError("expected a finite nonempty KN matrix")
    k, n = w.shape
    padded = np.zeros((n, ((k + 15) // 16) * 16), np.float32)
    padded[:, :k] = w.T
    groups = padded.reshape(n, -1, 16)
    maximum = np.max(np.abs(groups), axis=2)
    raw = np.where(maximum == 0, 1., maximum / 7.)
    scales = raw.astype(ml_dtypes.bfloat16)
    sf = scales.astype(np.float32)
    if not np.isfinite(sf).all() or (sf < np.finfo(np.float32).tiny).any():
        raise ValueError("scale is not representable as normal BF16")
    q = np.clip(np.rint(groups / sf[..., None]), -7, 7).astype(np.int8).reshape(n, -1)
    u = q.astype(np.uint8) & 15
    return u[:, ::2] | (u[:, 1::2] << 4), scales

def dequantize(packed, scales, k):
    import numpy as np
    p = np.asarray(packed, np.uint8)
    q = np.empty((p.shape[0], p.shape[1] * 2), np.int8)
    q[:, ::2] = (p & 15).astype(np.int8)
    q[:, 1::2] = (p >> 4).astype(np.int8)
    q[q >= 8] -= 16
    w = q.reshape(q.shape[0], -1, 16).astype(np.float32) * scales.astype(np.float32)[..., None]
    # The GEMM consumes BF16 dequantized weights, not exact FP32 products.
    return w.astype(scales.dtype).astype(np.float32).reshape(q.shape)[:, :k].T.copy()
