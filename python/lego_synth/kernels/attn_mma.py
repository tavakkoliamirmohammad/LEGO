"""FlashAttention-2 style forward pass through the CASTLE pipeline (f16 in, f32 or f16 out).

Q, K, V: [BH, N, D] row-major f16 (flat buffers); O: [BH, N, D] f32.
Block = BLOCK_M query rows, BLOCK_M / rows_per_warp warps of 16 or 32 rows. K/V tiles of BLOCK_N keys stream through an
S-stage cp.async pipeline in shared memory whose layouts are LEGO layouts (the synthesized
swizzle for the head dimension). S = Q K^T and O += P V use mma.sync m16n8k16; online softmax
uses exp2 with the scale folded in and quad shuffles for row max/sum.
"""
from lego.core import Row
from lego.backend.compiler import DType
from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer
from lego.mlir.ir import (Type, VectorType, F16Type, F32Type, DenseElementsAttr, IndexType, IntegerAttr,
                          FloatAttr, Operation, StringAttr)
from lego.mlir.dialects import arith as arith_d, memref as memref_d, nvgpu as nvgpu_d, vector as vector_d

from .gemm_mma import _idx, _add, _mul, _div, _rem, sA_layout_numel, pack_pair
from .addr import LaneTile


def _f32(v: float):
    return arith_d.ConstantOp(F32Type.get(), FloatAttr.get(F32Type.get(), float(v))).result


def _exp2(x):
    """Bare ex2.approx.ftz.f32 (one MUFU). math.exp2 lowers to libdevice's __nv_exp2f, and even the
    non-ftz ex2.approx.f32 makes ptxas add a compare and two multiplies per call for denormal
    results; the softmax never needs them (Triton and FlashAttention flush to zero here too)."""
    return Operation.create("llvm.inline_asm", results=[x.type], operands=[x],
                            attributes={"asm_string": StringAttr.get("ex2.approx.ftz.f32 $0, $1;"),
                                        "constraints": StringAttr.get("=f,f")}).result


def _fma(a, b, c):
    return Operation.create("math.fma", results=[a.type], operands=[a, b, c]).result


def build_attention(BH: int, N: int, D: int, sK_layout, sV_layout, BLOCK_M: int = 64, BLOCK_N: int = 32,
                    stages: int = 2, scale: float = None, name: str = "attn_mma", dynamic_shared: bool = None,
                    causal: bool = False, rows_per_warp: int = 16, out_dtype: str = "f32",
                    q_in_regs: bool = True, sQ_layout=None, delta_addr: bool = True) -> KernelBuilder:
    """causal=True applies the lower-triangular mask (key <= query) and skips key blocks past the
    query block (BLOCK_M must be a multiple of BLOCK_N so a query block's key range ends on a block boundary).
    rows_per_warp=32 gives each warp two 16-row m-tiles, so every K/V fragment load feeds four mma
    instead of two (halves shared-memory traffic per mma at the cost of registers).
    out_dtype="f16" writes O as f16 (what the library kernels return); "f32" keeps the accumulator type.
    q_in_regs=False re-reads the Q fragments from shared memory for every key block instead of pinning
    them in registers (FlashAttention-2's choice at large head dimensions).
    sQ_layout: shared layout of the Q tile (default row-major; pass a synthesized layout to make the
    one-time Q fragment loads conflict-free too).
    delta_addr: address every tile access as (per-lane base ^ K) + C (see kernels/addr.py) instead of
    evaluating the layout per access; False keeps one hoisted address per access (ablation)."""
    assert N % BLOCK_M == 0 and N % BLOCK_N == 0 and D % 16 == 0 and BLOCK_N % 16 == 0
    assert rows_per_warp in (16, 32) and BLOCK_M % rows_per_warp == 0
    assert out_dtype in ("f32", "f16")
    MT = rows_per_warp // 16
    warps = BLOCK_M // rows_per_warp
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
    O_layout = Row(BH * N, D // 2)      # O is stored in packed column pairs (one 32-/64-bit word each)
    fpK, fpV = sA_layout_numel(sK_layout, BLOCK_N, D), sA_layout_numel(sV_layout, BLOCK_N, D)
    if sQ_layout is None:
        sQ_layout = Row(BLOCK_M, D)
    fpQ = sA_layout_numel(sQ_layout, BLOCK_M, D)

    bufs = [LayoutBuffer(QKV_layout, shape=(BH * N * D,), dtype=DType.f16),   # Q
            LayoutBuffer(QKV_layout, shape=(BH * N * D,), dtype=DType.f16),   # K
            LayoutBuffer(QKV_layout, shape=(BH * N * D,), dtype=DType.f16),   # V
            LayoutBuffer(O_layout, shape=(BH * N * D // 2,), dtype=DType.i64 if out_dtype == "f32" else DType.i32),   # O (packed pairs)
            LayoutBuffer(sQ_layout, shape=(fpQ,), dtype=DType.f16, shared=True),
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
        wrow = _add(row_base, _mul(warp, _idx(rows_per_warp)))             # this warp's first query row
        kv_base = _mul(bh, _idx(N))
        chunks_per_row = D // 8
        # Lane -> tile coordinate mappings, symbolic (MLIR) and numeric (for verifying the delta rewrite).
        tK = LaneTile(ctx, sK_layout, _add(lane8, kb_hi), kb_d,
                      lambda t: (t % 32) % 8 + ((t % 32) // 16) * 8, lambda t: (((t % 32) // 8) % 2) * 8, threads, delta_addr)
        tV = LaneTile(ctx, sV_layout, lane16, lane_hi, lambda t: (t % 32) % 16, lambda t: ((t % 32) // 16) * 8, threads, delta_addr)
        tQ = LaneTile(ctx, sQ_layout, _add(_mul(warp, _idx(rows_per_warp)), lane16), lane_hi,
                      lambda t: (t // 32) * rows_per_warp + (t % 32) % 16, lambda t: ((t % 32) // 16) * 8, threads, delta_addr)
        cp_row, cp_col = _div(tid, _idx(chunks_per_row)), _mul(_rem(tid, _idx(chunks_per_row)), _idx(8))
        cK = LaneTile(ctx, sK_layout, cp_row, cp_col, lambda t: t // chunks_per_row, lambda t: (t % chunks_per_row) * 8, threads, delta_addr)
        cV = LaneTile(ctx, sV_layout, cp_row, cp_col, lambda t: t // chunks_per_row, lambda t: (t % chunks_per_row) * 8, threads, delta_addr)
        cQ = LaneTile(ctx, sQ_layout, cp_row, cp_col, lambda t: t // chunks_per_row, lambda t: (t % chunks_per_row) * 8, threads, delta_addr)

        def copy_dst(tile, it, total, c):
            """Shared destination of chunk `c` = tid + it*threads: a row delta when the chunk grid divides evenly."""
            if (it + 1) * threads <= total and threads % chunks_per_row == 0:
                return tile.at(it * threads // chunks_per_row, 0)
            r, c8 = _div(c, _idx(chunks_per_row)), _mul(_rem(c, _idx(chunks_per_row)), _idx(8))
            return ctx.apply(tile.layout, r, c8)

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
            dst = copy_dst(cQ, it, chunks_q, c)
            toks.append(nvgpu_d.device_async_copy(sQ, [dst], Q, [src], 8, results=[token_ty]))
        grp = nvgpu_d.device_async_create_group(toks, results=[token_ty])
        nvgpu_d.device_async_wait(grp, num_groups=0)
        ctx.barrier()
        def load_q_frag(mt, kk, tile=tQ):
            return nvgpu_d.ldmatrix(vA, sQ, [tile.at(mt * 16, kk * 16)], False, 4)

        q_regs = [[load_q_frag(mt, kk) for kk in range(DT)] for mt in range(MT)] if q_in_regs else None

        def issue_kv(kt_idx, stage, tiles=(cK, cV)):
            key0 = _add(kv_base, _mul(kt_idx, _idx(BLOCK_N)))
            offK, offV = _mul(stage, _idx(fpK)), _mul(stage, _idx(fpV))
            toks = []
            for it in range(iters_kv):
                c = chunk(it, chunks_kv)
                r, c8 = _div(c, _idx(D // 8)), _mul(_rem(c, _idx(D // 8)), _idx(8))
                src = ctx.apply(QKV_layout, _add(key0, r), c8)
                toks.append(nvgpu_d.device_async_copy(sK, [_add(offK, copy_dst(tiles[0], it, chunks_kv, c))], K, [src], 8, results=[token_ty]))
                toks.append(nvgpu_d.device_async_copy(sV, [_add(offV, copy_dst(tiles[1], it, chunks_kv, c))], V, [src], 8, results=[token_ty]))
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

        def step(stage, o_acc, m_i, l_i, kt_cur, tiles=(tK, tV, tQ)):
            tk, tv, tq = tiles
            offK, offV = _mul(stage, _idx(fpK)), _mul(stage, _idx(fpV))
            # S = Q K^T over this key block: NT_S n-tiles, accumulate over DT k-steps.
            q_frags = q_regs if q_in_regs else [[load_q_frag(mt, kk, tq) for kk in range(DT)] for mt in range(MT)]
            s_acc = [[zero for _ in range(NT_S)] for _ in range(MT)]
            for kk in range(DT):
                for nt2 in range(NT_S // 2):
                    k4 = nvgpu_d.ldmatrix(vA, sK, [_add(offK, tk.at(nt2 * 16, kk * 16))], False, 4)
                    b0 = vector_d.extract_strided_slice(vB, k4, [0, 0], [2, 2], [1, 1])
                    b1 = vector_d.extract_strided_slice(vB, k4, [2, 0], [2, 2], [1, 1])
                    for mt in range(MT):
                        s_acc[mt][2 * nt2] = nvgpu_d.mma_sync(vC, q_frags[mt][kk], b0, s_acc[mt][2 * nt2], [16, 8, 16])
                        s_acc[mt][2 * nt2 + 1] = nvgpu_d.mma_sync(vC, q_frags[mt][kk], b1, s_acc[mt][2 * nt2 + 1], [16, 8, 16])
            # Raw scores; the log2-domain scale is applied once to the row max and fused into the
            # exponent (p = exp2(s * sc - m)), as FlashAttention-2 and Triton do.
            sc = _f32(scale_log2)
            sv = [[[[vector_d.extract(s_acc[mt][nt], [], [i, j]) for j in range(2)] for i in range(2)]
                   for nt in range(NT_S)] for mt in range(MT)]
            if causal:
                # Mask keys after the query: score -> -1e30 when key > query (only the diagonal
                # key blocks can be partially masked; earlier blocks are fully visible).
                neg = _f32(-1e30)
                for mt in range(MT):
                    for nt in range(NT_S):
                        for i in range(2):
                            qrow = _add(_add(_mul(mblk, _idx(BLOCK_M)),
                                             _add(_mul(warp, _idx(rows_per_warp)), _idx(mt * 16 + 8 * i))), g_row)
                            for j in range(2):
                                key = _add(_add(_mul(kt_cur, _idx(BLOCK_N)), _idx(nt * 8 + j)), g_col2)
                                visible = arith_d.CmpIOp(arith_d.CmpIPredicate.ule, key, qrow).result
                                sv[mt][nt][i][j] = arith_d.SelectOp(visible, sv[mt][nt][i][j], neg).result
            m_new = [[None] * 2 for _ in range(MT)]; alpha = [[None] * 2 for _ in range(MT)]
            l_new = [[None] * 2 for _ in range(MT)]
            p = [[[None] * NT_S for _ in range(2)] for _ in range(MT)]
            for mt in range(MT):
                for i in range(2):
                    mx = sv[mt][0][i][0]
                    for nt in range(NT_S):
                        for j in range(2):
                            mx = fmax(mx, sv[mt][nt][i][j])
                    mx = quad_reduce(mx, fmax)
                    mn = fmax(m_i[mt][i], fmul(mx, sc))
                    m_new[mt][i] = mn
                    alpha[mt][i] = _exp2(fsub(m_i[mt][i], mn))
                    neg_mn = arith_d.NegFOp(mn).result
                    rs = _f32(0.0)
                    for nt in range(NT_S):
                        pe = [_exp2(_fma(sv[mt][nt][i][j], sc, neg_mn)) for j in range(2)]
                        p[mt][i][nt] = pe
                        rs = fadd(rs, fadd(pe[0], pe[1]))
                    # l stays a per-lane partial sum (each lane holds 2 of the row's 8*NT_S scores);
                    # it is reduced across the quad once, in the epilogue.
                    l_new[mt][i] = fadd(fmul(l_i[mt][i], alpha[mt][i]), rs)
            # Rescale O rows by alpha and add P V.
            o_new = [[None] * DN for _ in range(MT)]
            for mt in range(MT):
                for dn in range(DN):
                    v = o_acc[mt][dn]
                    for i in range(2):
                        for j in range(2):
                            e = fmul(vector_d.extract(v, [], [i, j]), alpha[mt][i])
                            v = vector_d.insert(e, v, [], [i, j])
                    o_new[mt][dn] = v
            # P as A fragments: k-step kk2 covers S n-tiles 2kk2 and 2kk2+1.
            p_frags = [[None] * KT_PV for _ in range(MT)]
            for mt in range(MT):
                pm = p[mt]
                for kk2 in range(KT_PV):
                    elems = [pm[0][2 * kk2][0], pm[0][2 * kk2][1], pm[1][2 * kk2][0], pm[1][2 * kk2][1],
                             pm[0][2 * kk2 + 1][0], pm[0][2 * kk2 + 1][1], pm[1][2 * kk2 + 1][0], pm[1][2 * kk2 + 1][1]]
                    frag = arith_d.ConstantOp(vA, DenseElementsAttr.get_splat(vA, FloatAttr.get(f16, 0.0))).result
                    for idx, e in enumerate(elems):
                        e16 = arith_d.TruncFOp(f16, e).result
                        frag = vector_d.insert(e16, frag, [], [idx // 2, idx % 2])
                    p_frags[mt][kk2] = frag
            for kk2 in range(KT_PV):
                for dn2 in range(DN // 2):
                    v4 = nvgpu_d.ldmatrix(vA, sV, [_add(offV, tv.at(kk2 * 16, dn2 * 16))], True, 4)
                    b0 = vector_d.extract_strided_slice(vB, v4, [0, 0], [2, 2], [1, 1])
                    b1 = vector_d.extract_strided_slice(vB, v4, [2, 0], [2, 2], [1, 1])
                    for mt in range(MT):
                        o_new[mt][2 * dn2] = nvgpu_d.mma_sync(vC, p_frags[mt][kk2], b0, o_new[mt][2 * dn2], [16, 8, 16])
                        o_new[mt][2 * dn2 + 1] = nvgpu_d.mma_sync(vC, p_frags[mt][kk2], b1, o_new[mt][2 * dn2 + 1], [16, 8, 16])
            return o_new, m_new, l_new

        o_acc = [[zero for _ in range(DN)] for _ in range(MT)]
        m_i = [[_f32(-1e30), _f32(-1e30)] for _ in range(MT)]
        l_i = [[_f32(0.0), _f32(0.0)] for _ in range(MT)]

        def flatten(o, m, l):
            return ([o[mt][dn] for mt in range(MT) for dn in range(DN)] + [m[mt][i] for mt in range(MT) for i in range(2)]
                    + [l[mt][i] for mt in range(MT) for i in range(2)])

        def unflatten(vals):
            vals = list(vals)
            o = [[vals[mt * DN + dn] for dn in range(DN)] for mt in range(MT)]
            base = MT * DN
            m = [[vals[base + mt * 2 + i] for i in range(2)] for mt in range(MT)]
            base += MT * 2
            l = [[vals[base + mt * 2 + i] for i in range(2)] for mt in range(MT)]
            return o, m, l

        if S == 1:
            for kt in range(KT):
                grp = issue_kv(_idx(kt), _idx(0))
                nvgpu_d.device_async_wait(grp, num_groups=0)
                ctx.barrier()
                o_acc, m_i, l_i = step(_idx(0), o_acc, m_i, l_i, _idx(kt))
                ctx.barrier()
        else:
            groups = [issue_kv(_idx(st), _idx(st)) for st in range(S - 1)]
            carried = flatten(o_acc, m_i, l_i)
            # Key blocks visible to this query block (causal: up to and including the diagonal).
            n_kv = _mul(_add(mblk, _idx(1)), _idx(BLOCK_M // BLOCK_N)) if causal else _idx(KT)
            assert KT % S == 0

            def loop_body(iv, *vals):
                # Unrolled by S: sub-iteration st handles key block iv*S + st in stage st, so stage
                # indices and every shared-memory offset are compile-time constants.
                o_cur, m_cur, l_cur = unflatten(vals)
                kt_base = _mul(iv, _idx(S))
                loop_tiles = (tK.in_loop(), tV.in_loop(), tQ.in_loop())
                loop_copy = (cK.in_loop(), cV.in_loop())
                for st in range(S):
                    nvgpu_d.device_async_wait(groups[-1], num_groups=S - 2)
                    ctx.barrier()
                    nxt = _add(kt_base, _idx(st + S - 1))
                    clamped = arith_d.MinUIOp(nxt, _idx(KT - 1)).result
                    issue_kv(clamped, _idx((st + S - 1) % S), loop_copy)
                    o_cur, m_cur, l_cur = step(_idx(st), o_cur, m_cur, l_cur, _add(kt_base, _idx(st)), loop_tiles)
                return flatten(o_cur, m_cur, l_cur)

            res = ctx.for_range(_div(n_kv, _idx(S)), loop_body, init_vals=carried)
            o_acc, m_i, l_i = unflatten(res)

        for mt in range(MT):
            inv_l = [arith_d.DivFOp(_f32(1.0), quad_reduce(l_i[mt][i], fadd)).result for i in range(2)]
            for dn in range(DN):
                for i in range(2):
                    pair = [fmul(vector_d.extract(o_acc[mt][dn], [], [i, j]), inv_l[i]) for j in range(2)]
                    r = _add(_add(wrow, _idx(mt * 16 + 8 * i)), g_row)
                    cc2 = _add(_idx(dn * 4), _rem(lane, _idx(4)))
                    memref_d.StoreOp(pack_pair(pair, out_dtype), O, [ctx.apply(O_layout, r, cc2)])

    smem_bytes = (fpQ + S * (fpK + fpV)) * 2
    if dynamic_shared is None:
        dynamic_shared = smem_bytes > 48 * 1024
    return KernelBuilder(buffers=bufs, kernel_body=body, name=name,
                         grid=(N // BLOCK_M, BH, 1), block=(threads, 1, 1), dynamic_shared=dynamic_shared)
