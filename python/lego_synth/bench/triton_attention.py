"""Triton fused attention baseline: the Triton tutorial forward kernel (block-pointer version of
`06-fused-attention.py`), used as-is for power-of-two head dimensions and padded to the next power of
two with zero-filled boundary loads for the others (Triton's `tl.arange` only accepts powers of two,
so this is what a Triton user has to do for D = 80 / 96 / 112).

Inputs are our [BH, N, D] f16 tensors (Z = 1, H = BH); output is f16 like the SDPA backends.
"""
import torch
import triton
import triton.language as tl


def head_dim_pad(d: int) -> int:
    return 1 << (d - 1).bit_length()


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    N_CTX: tl.constexpr, PAD: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if PAD:
            k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        else:
            k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        if PAD:
            v = tl.load(V_block_ptr, boundary_check=(1,), padding_option="zero")
        else:
            v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,
              stride_qh, stride_qm, stride_qk,
              stride_kh, stride_kn, stride_kk,
              stride_vh, stride_vk, stride_vn,
              stride_oh, stride_om, stride_on,
              H, N_CTX, D_REAL,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
              STAGE: tl.constexpr, PAD: tl.constexpr):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    qvk_offset = off_h.to(tl.int64) * stride_qh
    Q_block_ptr = tl.make_block_ptr(base=Q + qvk_offset, shape=(N_CTX, D_REAL), strides=(stride_qm, stride_qk),
                                    offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(base=V + off_h.to(tl.int64) * stride_vh, shape=(N_CTX, D_REAL),
                                    strides=(stride_vk, stride_vn), offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM),
                                    order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + off_h.to(tl.int64) * stride_kh, shape=(D_REAL, N_CTX),
                                    strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N),
                                    order=(0, 1))
    O_block_ptr = tl.make_block_ptr(base=Out + off_h.to(tl.int64) * stride_oh, shape=(N_CTX, D_REAL),
                                    strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0),
                                    block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    if PAD:
        q = tl.load(Q_block_ptr, boundary_check=(1,), padding_option="zero")
    else:
        q = tl.load(Q_block_ptr)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N, 4 - STAGE, offs_m, offs_n, N_CTX, PAD)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N, 2, offs_m, offs_n, N_CTX, PAD)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_h * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    if PAD:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(1,))
    else:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


_CONFIGS = [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_stages=s, num_warps=w)
            for bm in (64, 128) for bn in (32, 64, 128) for s in (2, 3, 4) for w in (4, 8)]


def _prune(configs, named_args, **kwargs):
    hd = kwargs.get("HEAD_DIM", named_args.get("HEAD_DIM"))   # constexpr args arrive as keywords
    keep = []
    for c in configs:
        bm, bn = c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"]
        if bn > hd:
            continue
        if bm * bn < 128 * 128 and c.num_warps == 8:
            continue
        keep.append(c)
    return keep


_attn_fwd_tuned = triton.autotune(configs=_CONFIGS, key=["N_CTX", "HEAD_DIM", "STAGE"],
                                  prune_configs_by={"early_config_prune": _prune})(_attn_fwd)


def _launch(kernel, q, k, v, causal, **meta):
    BH, N, D = q.shape
    hd = head_dim_pad(D)
    o = torch.empty_like(q)
    M = torch.empty((BH, N), device=q.device, dtype=torch.float32)
    stage = 3 if causal else 1
    grid = lambda args: (triton.cdiv(N, args["BLOCK_M"]), BH, 1)
    kernel[grid](q, k, v, 1.0 / D ** 0.5, M, o,
                 q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                 v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
                 BH, N, D, HEAD_DIM=hd, STAGE=stage, PAD=(hd != D), **meta)
    return o


def triton_attention(q, k, v, causal=False, config=None):
    """Fixed configuration (default: the tutorial's 64x64, 4 warps, 3 stages)."""
    cfg = dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=3) if config is None else dict(config)
    if cfg["BLOCK_N"] > head_dim_pad(q.shape[-1]):
        cfg["BLOCK_N"] = head_dim_pad(q.shape[-1])
    return _launch(_attn_fwd, q, k, v, causal, **cfg)


def triton_attention_tuned(q, k, v, causal=False):
    """Autotuned over block sizes, stages and warps (the tutorial's search space widened to BLOCK_N=32)."""
    return _launch(_attn_fwd_tuned, q, k, v, causal)
