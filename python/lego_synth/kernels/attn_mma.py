"""FlashAttention-2 style forward pass through the CASTLE pipeline (non-causal, f16 in, f32 out).

Q, K, V: [BH, N, D] row-major f16 (flat buffers); O: [BH, N, D] f32.
Block = BLOCK_M query rows, 4 warps x 16 rows. K/V tiles of BLOCK_N keys stream through an
S-stage cp.async pipeline in shared memory whose layouts are LEGO layouts (the synthesized
swizzle for the head dimension). S = Q K^T and O += P V use mma.sync m16n8k16; online softmax
uses exp2 with the scale folded in and quad shuffles for row max/sum.
"""
from lego.core import Row
from lego.backend.compiler import DType
from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer
from lego.mlir.ir import (Type, VectorType, F16Type, F32Type, DenseElementsAttr, IndexType, IntegerAttr,
                          FloatAttr, Operation)
from lego.mlir.dialects import arith as arith_d, memref as memref_d, nvgpu as nvgpu_d, vector as vector_d

from .gemm_mma import _idx, _add, _mul, _div, _rem, sA_layout_numel


def _f32(v: float):
    return arith_d.ConstantOp(F32Type.get(), FloatAttr.get(F32Type.get(), float(v))).result


def _exp2(x):
    return Operation.create("math.exp2", results=[x.type], operands=[x]).result


def build_attention(BH: int, N: int, D: int, sK_layout, sV_layout, BLOCK_M: int = 64, BLOCK_N: int = 32,
                    stages: int = 2, scale: float = None, name: str = "attn_mma", dynamic_shared: bool = None,
                    causal: bool = False) -> KernelBuilder:
    """causal=True applies the lower-triangular mask (key <= query) and skips key blocks past the
    query block (BLOCK_M must be a multiple of BLOCK_N so a query block's key range ends on a block boundary)."""
    assert N % BLOCK_M == 0 and N % BLOCK_N == 0 and D % 16 == 0 and BLOCK_N % 16 == 0
    warps = BLOCK_M // 16
    threads = 32 * warps
    KT = N // BLOCK_N
    S = max(1, stages)
    assert S <= KT
    if causal:
        assert BLOCK_M % BLOCK_N == 0
        assert S == 1 or (BLOCK_M // BLOCK_N) % S == 0, "per-query-block key count must be a multiple of the pipeline depth"
    DT, DN = D // 16, D // 8            # k-steps over D for QK^T; n-tiles over D for O
    NT_S = BLOCK_N // 8                 # n-tiles of S per warp (keys)
    KT_PV = BLOCK_N // 16               # k-steps over keys for PV
    scale = (1.0 / D ** 0.5) if scale is None else scale
    scale_log2 = scale * 1.4426950408889634
    QKV_layout = Row(BH * N, D)         # [bh*N + n, d]
    fpK, fpV = sA_layout_numel(sK_layout, BLOCK_N, D), sA_layout_numel(sV_layout, BLOCK_N, D)
    fpQ = BLOCK_M * D

    bufs = [LayoutBuffer(QKV_layout, shape=(BH * N * D,), dtype=DType.f16),   # Q
            LayoutBuffer(QKV_layout, shape=(BH * N * D,), dtype=DType.f16),   # K
            LayoutBuffer(QKV_layout, shape=(BH * N * D,), dtype=DType.f16),   # V
            LayoutBuffer(QKV_layout, shape=(BH * N * D,), dtype=DType.f32),   # O
            LayoutBuffer(Row(BLOCK_M, D), shape=(fpQ,), dtype=DType.f16, shared=True),
            LayoutBuffer(sK_layout, shape=(S * fpK,), dtype=DType.f16, shared=True),
            LayoutBuffer(sV_layout, shape=(S * fpV,), dtype=DType.f16, shared=True)]

    def body(ctx):
        Q, K, V, O, sQ, sK, sV = ctx._buf_vals
        f16, f32 = F16Type.get(), F32Type.get()
        vA = VectorType.get([4, 2], f16); vB = VectorType.get([2, 2], f16)
        vC = VectorType.get([2, 2], f32); vC16 = VectorType.get([2, 2], f16)
        token_ty = Type.parse("!nvgpu.device.async.token")
        tid = ctx.thread_id.x
        mblk, bh = ctx.block_id.x, ctx.block_id.y
        c32 = _idx(32)
        warp, lane = _div(tid, c32), _rem(tid, c32)
        lane16, lane_hi = _rem(lane, _idx(16)), _mul(_div(lane, _idx(16)), _idx(8))
        lane8 = _rem(lane, _idx(8))
        kb_hi = _mul(_div(lane, _idx(16)), _idx(8))                      # +8 keys for lanes 16..31
        kb_d = _mul(_rem(_div(lane, _idx(8)), _idx(2)), _idx(8))         # +8 d for lanes 8..15, 24..31
        g_row, g_col2 = _div(lane, _idx(4)), _mul(_rem(lane, _idx(4)), _idx(2))
        row_base = _add(_mul(bh, _idx(N)), _mul(mblk, _idx(BLOCK_M)))    # first query row (global row index)
        wrow = _add(row_base, _mul(warp, _idx(16)))                        # this warp's first query row
        kv_base = _mul(bh, _idx(N))

        zero = arith_d.ConstantOp(vC, DenseElementsAttr.get_splat(vC, FloatAttr.get(f32, 0.0))).result
        chunks_q, chunks_kv = BLOCK_M * D // 8, BLOCK_N * D // 8
        # Chunk counts need not be multiples of the thread count: the ragged tail clamps
        # its chunk index (a few lanes re-copy the last chunk, which is harmless).
        iters_q, iters_kv = -(-chunks_q // threads), -(-chunks_kv // threads)

        def chunk(it, total):
            c = _add(tid, _idx(it * threads))
            return arith_d.MinUIOp(c, _idx(total - 1)).result if (it + 1) * threads > total else c

        # ---- Q tile -> shared -> A fragments in registers (one-time) ----
        toks = []
        for it in range(iters_q):
            c = chunk(it, chunks_q)
            r, c8 = _div(c, _idx(D // 8)), _mul(_rem(c, _idx(D // 8)), _idx(8))
            src = ctx.apply(QKV_layout, _add(row_base, r), c8)
            dst = _add(_mul(r, _idx(D)), c8)
            toks.append(nvgpu_d.device_async_copy(sQ, [dst], Q, [src], 8, results=[token_ty]))
        grp = nvgpu_d.device_async_create_group(toks, results=[token_ty])
        nvgpu_d.device_async_wait(grp, num_groups=0)
        ctx.barrier()
        q_frags = []
        for kk in range(DT):
            r = _add(_mul(warp, _idx(16)), lane16)
            cidx = _add(_idx(kk * 16), lane_hi)
            q_frags.append(nvgpu_d.ldmatrix(vA, sQ, [_add(_mul(r, _idx(D)), cidx)], False, 4))

        def issue_kv(kt_idx, stage):
            key0 = _add(kv_base, _mul(kt_idx, _idx(BLOCK_N)))
            offK, offV = _mul(stage, _idx(fpK)), _mul(stage, _idx(fpV))
            toks = []
            for it in range(iters_kv):
                c = chunk(it, chunks_kv)
                r, c8 = _div(c, _idx(D // 8)), _mul(_rem(c, _idx(D // 8)), _idx(8))
                src = ctx.apply(QKV_layout, _add(key0, r), c8)
                toks.append(nvgpu_d.device_async_copy(sK, [_add(offK, ctx.apply(sK_layout, r, c8))], K, [src], 8, results=[token_ty]))
                toks.append(nvgpu_d.device_async_copy(sV, [_add(offV, ctx.apply(sV_layout, r, c8))], V, [src], 8, results=[token_ty]))
            return nvgpu_d.device_async_create_group(toks, results=[token_ty])

        def quad_reduce(v, op):
            for m in (1, 2):
                s = ctx.shuffle_xor(v, m)
                v = op(v, s)
            return v

        fmax = lambda a, b: arith_d.MaximumFOp(a, b).result
        fadd = lambda a, b: arith_d.AddFOp(a, b).result
        fmul = lambda a, b: arith_d.MulFOp(a, b).result
        fsub = lambda a, b: arith_d.SubFOp(a, b).result

        def step(stage, o_acc, m_i, l_i, kt_cur):
            offK, offV = _mul(stage, _idx(fpK)), _mul(stage, _idx(fpV))
            # S = Q K^T over this key block: NT_S n-tiles, accumulate over DT k-steps.
            s_acc = [zero for _ in range(NT_S)]
            for kk in range(DT):
                for nt2 in range(NT_S // 2):
                    r = _add(_add(_idx(nt2 * 16), lane8), kb_hi)
                    cidx = _add(_idx(kk * 16), kb_d)
                    k4 = nvgpu_d.ldmatrix(vA, sK, [_add(offK, ctx.apply(sK_layout, r, cidx))], False, 4)
                    b0 = vector_d.extract_strided_slice(vB, k4, [0, 0], [2, 2], [1, 1])
                    b1 = vector_d.extract_strided_slice(vB, k4, [2, 0], [2, 2], [1, 1])
                    s_acc[2 * nt2] = nvgpu_d.mma_sync(vC, q_frags[kk], b0, s_acc[2 * nt2], [16, 8, 16])
                    s_acc[2 * nt2 + 1] = nvgpu_d.mma_sync(vC, q_frags[kk], b1, s_acc[2 * nt2 + 1], [16, 8, 16])
            # Scale (log2 domain) and row max over this block for the lane's two rows.
            sc = _f32(scale_log2)
            sv = [[[fmul(vector_d.extract(s_acc[nt], [], [i, j]), sc) for j in range(2)] for i in range(2)] for nt in range(NT_S)]
            if causal:
                # Mask keys after the query: score -> -1e30 when key > query (only the diagonal
                # key blocks can be partially masked; earlier blocks are fully visible).
                neg = _f32(-1e30)
                for nt in range(NT_S):
                    for i in range(2):
                        qrow = _add(_add(_mul(mblk, _idx(BLOCK_M)), _add(_mul(warp, _idx(16)), _idx(8 * i))), g_row)
                        for j in range(2):
                            key = _add(_add(_mul(kt_cur, _idx(BLOCK_N)), _idx(nt * 8 + j)), g_col2)
                            visible = arith_d.CmpIOp(arith_d.CmpIPredicate.ule, key, qrow).result
                            sv[nt][i][j] = arith_d.SelectOp(visible, sv[nt][i][j], neg).result
            m_new, alpha, p, rowsum = [], [], [[None] * NT_S for _ in range(2)], []
            for i in range(2):
                mx = sv[0][i][0]
                for nt in range(NT_S):
                    for j in range(2):
                        mx = fmax(mx, sv[nt][i][j])
                mx = quad_reduce(mx, fmax)
                mn = fmax(m_i[i], mx)
                m_new.append(mn)
                alpha.append(_exp2(fsub(m_i[i], mn)))
                rs = _f32(0.0)
                for nt in range(NT_S):
                    pe = [_exp2(fsub(sv[nt][i][j], mn)) for j in range(2)]
                    p[i][nt] = pe
                    rs = fadd(rs, fadd(pe[0], pe[1]))
                rowsum.append(quad_reduce(rs, fadd))
            l_new = [fadd(fmul(l_i[i], alpha[i]), rowsum[i]) for i in range(2)]
            # Rescale O rows by alpha and add P V.
            o_new = []
            for dn in range(DN):
                v = o_acc[dn]
                for i in range(2):
                    for j in range(2):
                        e = fmul(vector_d.extract(v, [], [i, j]), alpha[i])
                        v = vector_d.insert(e, v, [], [i, j])
                o_new.append(v)
            # P as A fragments: k-step kk2 covers S n-tiles 2kk2 and 2kk2+1.
            p_frags = []
            for kk2 in range(KT_PV):
                elems = [p[0][2 * kk2][0], p[0][2 * kk2][1], p[1][2 * kk2][0], p[1][2 * kk2][1],
                         p[0][2 * kk2 + 1][0], p[0][2 * kk2 + 1][1], p[1][2 * kk2 + 1][0], p[1][2 * kk2 + 1][1]]
                frag = arith_d.ConstantOp(vA, DenseElementsAttr.get_splat(vA, FloatAttr.get(f16, 0.0))).result
                for idx, e in enumerate(elems):
                    e16 = arith_d.TruncFOp(f16, e).result
                    frag = vector_d.insert(e16, frag, [], [idx // 2, idx % 2])
                p_frags.append(frag)
            for kk2 in range(KT_PV):
                for dn2 in range(DN // 2):
                    r = _add(_idx(kk2 * 16), lane16)
                    cidx = _add(_idx(dn2 * 16), lane_hi)
                    v4 = nvgpu_d.ldmatrix(vA, sV, [_add(offV, ctx.apply(sV_layout, r, cidx))], True, 4)
                    b0 = vector_d.extract_strided_slice(vB, v4, [0, 0], [2, 2], [1, 1])
                    b1 = vector_d.extract_strided_slice(vB, v4, [2, 0], [2, 2], [1, 1])
                    o_new[2 * dn2] = nvgpu_d.mma_sync(vC, p_frags[kk2], b0, o_new[2 * dn2], [16, 8, 16])
                    o_new[2 * dn2 + 1] = nvgpu_d.mma_sync(vC, p_frags[kk2], b1, o_new[2 * dn2 + 1], [16, 8, 16])
            return o_new, m_new, l_new

        o_acc = [zero for _ in range(DN)]
        m_i = [_f32(-1e30), _f32(-1e30)]
        l_i = [_f32(0.0), _f32(0.0)]
        if S == 1:
            for kt in range(KT):
                grp = issue_kv(_idx(kt), _idx(0))
                nvgpu_d.device_async_wait(grp, num_groups=0)
                ctx.barrier()
                o_acc, m_i, l_i = step(_idx(0), o_acc, m_i, l_i, _idx(kt))
                ctx.barrier()
        else:
            groups = [issue_kv(_idx(st), _idx(st)) for st in range(S - 1)]
            carried = o_acc + m_i + l_i

            def loop_body(iv, *vals):
                o_cur, m_cur, l_cur = list(vals[:DN]), list(vals[DN:DN + 2]), list(vals[DN + 2:DN + 4])
                nvgpu_d.device_async_wait(groups[-1], num_groups=S - 2)
                ctx.barrier()
                nxt = _add(iv, _idx(S - 1))
                clamped = arith_d.MinUIOp(nxt, _idx(KT - 1)).result
                issue_kv(clamped, _rem(nxt, _idx(S)))
                o_n, m_n, l_n = step(_rem(iv, _idx(S)), o_cur, m_cur, l_cur, iv)
                return o_n + m_n + l_n

            # Causal: a query block only needs key blocks up to its own diagonal.
            n_kv = _add(_mul(_add(mblk, _idx(1)), _idx(BLOCK_M // BLOCK_N)), _idx(0)) if causal else _idx(KT)
            if causal:
                # Clamp prologue and loop refills to the visible range so no block reads past it.
                pass
            res = ctx.for_range(n_kv, loop_body, init_vals=carried)
            o_acc, m_i, l_i = list(res[:DN]), list(res[DN:DN + 2]), list(res[DN + 2:DN + 4])

        inv_l = [arith_d.DivFOp(_f32(1.0), l_i[i]).result for i in range(2)]
        for dn in range(DN):
            for i in range(2):
                for j in range(2):
                    e = fmul(vector_d.extract(o_acc[dn], [], [i, j]), inv_l[i])
                    r = _add(_add(wrow, _idx(8 * i)), g_row)
                    cc = _add(_idx(dn * 8 + j), g_col2)
                    memref_d.StoreOp(e, O, [ctx.apply(QKV_layout, r, cc)])

    smem_bytes = (fpQ + S * (fpK + fpV)) * 2
    if dynamic_shared is None:
        dynamic_shared = smem_bytes > 48 * 1024
    return KernelBuilder(buffers=bufs, kernel_body=body, name=name,
                         grid=(N // BLOCK_M, BH, 1), block=(threads, 1, 1), dynamic_shared=dynamic_shared)
