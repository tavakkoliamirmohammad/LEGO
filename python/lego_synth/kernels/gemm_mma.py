"""Tensor-core GEMM (C = A @ B, f16 in, f32 accumulate) through the CASTLE pipeline.

Every shared-memory address goes through `lego.apply` on the tile's LEGO layout, so the
synthesized swizzle is the only thing that changes between layout variants:
  global -> shared : cp.async, 16 B per lane, dense row-major chunk assignment
  shared -> tensor : ldmatrix.x4 (A, row-major) and ldmatrix.x2.trans (B, row-major K x N)
  compute          : mma.sync.m16n8k16 f16 -> f32, warps tiled warps_m x warps_n
v4: scf.for K loop unrolled by the pipeline depth S (constant stage indices), S-stage cp.async pipeline (wait_group S-2, refill the stage computed last sub-iteration); B via ldmatrix.x4.trans.
"""
from lego.core import Row
from lego.backend.compiler import DType
from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer
from lego.mlir.ir import (Type, VectorType, F16Type, F32Type, DenseElementsAttr, IndexType,
                          IntegerAttr, IntegerType, FloatAttr)
from lego.mlir.dialects import arith as arith_d, memref as memref_d, nvgpu as nvgpu_d, vector as vector_d, gpu as gpu_d
from .addr import LaneTile


def _idx(v: int):
    return arith_d.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), int(v))).result


def _add(a, b): return arith_d.AddIOp(a, b).result
def _mul(a, b): return arith_d.MulIOp(a, b).result
def _div(a, b): return arith_d.DivUIOp(a, b).result
def _rem(a, b): return arith_d.RemUIOp(a, b).result


def build_gemm(M: int, N: int, K: int, BM: int, BN: int, BK: int, sA_layout, sB_layout,
               warps_m: int = 2, warps_n: int = 2, name: str = "gemm_mma", unroll_k: bool = False,
               stages: int = 3, dynamic_shared: bool = None, prefetch_frags: bool = False,
               group_m: int = 0, out_dtype: str = "f32", delta_addr: bool = True, packed_out: bool = True) -> KernelBuilder:
    """group_m > 0 remaps block ids so consecutive blocks share A row-tiles (L2-friendly grouped ordering).
    stages > 1 enables a multi-stage cp.async pipeline (unroll_k forces the single-stage form).
    out_dtype="f16" stores C as f16 (what cuBLAS returns for f16 inputs); accumulation stays f32.
    packed_out: the epilogue stores each fragment's two adjacent columns as one packed word (i32 for
    f16 pairs, i64 for f32 pairs) and C is declared as a buffer of M*N/2 words; False keeps scalar
    stores into an M*N buffer of the element type (mlir-runner tests print that form).
    delta_addr: address tile accesses as (per-lane base ^ K) + C (kernels/addr.py) instead of one hoisted
    address per access."""
    assert M % BM == 0 and N % BN == 0 and K % BK == 0 and BK % 16 == 0
    assert out_dtype in ("f32", "f16")
    WM, WN = BM // warps_m, BN // warps_n
    assert WM % 16 == 0 and WN % 16 == 0, "B fragments are loaded two n-tiles at a time (ldmatrix.x4.trans)"
    threads = 32 * warps_m * warps_n
    MT, NT = WM // 16, WN // 8
    KT = K // BK
    S = 1 if unroll_k else max(1, stages)
    assert S <= KT
    A_layout, B_layout = Row(M, K), Row(K, N)
    C_layout = Row(M, N // 2) if packed_out else Row(M, N)   # packed: C addressed in column pairs
    fpA, fpB = sA_layout_numel(sA_layout, BM, BK), sA_layout_numel(sB_layout, BK, BN)
    if packed_out:
        C_buf = LayoutBuffer(C_layout, shape=(M * N // 2,), dtype=DType.i64 if out_dtype == "f32" else DType.i32)
    else:
        C_buf = LayoutBuffer(C_layout, shape=(M * N,), dtype=DType.f32 if out_dtype == "f32" else DType.f16)

    bufs = [LayoutBuffer(A_layout, shape=(M * K,), dtype=DType.f16),
            LayoutBuffer(B_layout, shape=(K * N,), dtype=DType.f16),
            C_buf,
            LayoutBuffer(sA_layout, shape=(S * fpA,), dtype=DType.f16, shared=True),
            LayoutBuffer(sB_layout, shape=(S * fpB,), dtype=DType.f16, shared=True)]

    def body(ctx):
        A, B, C, sA, sB = ctx._buf_vals
        f16, f32 = F16Type.get(), F32Type.get()
        vA = VectorType.get([4, 2], f16); vB = VectorType.get([2, 2], f16); vC = VectorType.get([2, 2], f32)
        token_ty = Type.parse("!nvgpu.device.async.token")
        tid = ctx.thread_id.x
        bx, by = ctx.block_id.x, ctx.block_id.y
        if group_m:
            # Linear program id over a (M/BM) x (N/BN) grid, walked in groups of group_m row-tiles
            # (the ordering Triton's matmul tutorial uses): blocks that run together share A tiles.
            n_tiles_m, n_tiles_n = M // BM, N // BN
            pid = _add(_mul(by, _idx(n_tiles_n)), bx)
            width = group_m * n_tiles_n
            gid = _div(pid, _idx(width))
            first_m = _mul(gid, _idx(group_m))
            rem_m = arith_d.SubIOp(_idx(n_tiles_m), first_m).result
            gsize = arith_d.MinUIOp(_idx(group_m), rem_m).result
            local = _rem(pid, _idx(width))
            by = _add(first_m, _rem(local, gsize))
            bx = _div(local, gsize)
        c32 = _idx(32)
        warp, lane = _div(tid, c32), _rem(tid, c32)
        wm, wn = _div(warp, _idx(warps_n)), _rem(warp, _idx(warps_n))
        row0, col0 = _mul(by, _idx(BM)), _mul(bx, _idx(BN))
        lane16, lane_hi = _rem(lane, _idx(16)), _mul(_div(lane, _idx(16)), _idx(8))   # ldmatrix lane -> (row, col) offsets
        g_row, g_col2 = _div(lane, _idx(4)), _mul(_rem(lane, _idx(4)), _idx(2))       # mma C fragment lane offsets

        zero = arith_d.ConstantOp(vC, DenseElementsAttr.get_splat(vC, FloatAttr.get(f32, 0.0))).result
        acc = [[zero for _ in range(NT)] for _ in range(MT)]

        chunks_A, chunks_B = BM * BK // 8, BK * BN // 8
        assert chunks_A % threads == 0 and chunks_B % threads == 0
        cprA, cprB = BK // 8, BN // 8   # 16-byte chunks per tile row

        # Lane -> tile coordinate mappings, symbolic (MLIR) and numeric (to verify the delta rewrite).
        tA = LaneTile(ctx, sA_layout, _add(_mul(wm, _idx(WM)), lane16), lane_hi,
                      lambda t: ((t // 32) // warps_n) * WM + (t % 32) % 16, lambda t: ((t % 32) // 16) * 8, threads, delta_addr)
        tB = LaneTile(ctx, sB_layout, lane16, _add(_mul(wn, _idx(WN)), lane_hi),
                      lambda t: (t % 32) % 16, lambda t: ((t // 32) % warps_n) * WN + ((t % 32) // 16) * 8, threads, delta_addr)
        cA = LaneTile(ctx, sA_layout, _div(tid, _idx(cprA)), _mul(_rem(tid, _idx(cprA)), _idx(8)),
                      lambda t: t // cprA, lambda t: (t % cprA) * 8, threads, delta_addr)
        cB = LaneTile(ctx, sB_layout, _div(tid, _idx(cprB)), _mul(_rem(tid, _idx(cprB)), _idx(8)),
                      lambda t: t // cprB, lambda t: (t % cprB) * 8, threads, delta_addr)

        def copy_dst(tile, cpr, it):
            if threads % cpr == 0:
                return tile.at(it * threads // cpr, 0)
            c = _add(tid, _idx(it * threads))
            return ctx.apply(tile.layout, _div(c, _idx(cpr)), _mul(_rem(c, _idx(cpr)), _idx(8)))

        def issue_loads(kt_idx, stage, tiles=(cA, cB)):
            """cp.async the A and B tiles of k-slice kt_idx into shared stage `stage`; returns tokens."""
            k0 = _mul(kt_idx, _idx(BK))
            offA, offB = _mul(stage, _idx(fpA)), _mul(stage, _idx(fpB))
            tokens = []
            for it in range(chunks_A // threads):
                c = _add(tid, _idx(it * threads))
                r, c8 = _div(c, _idx(cprA)), _mul(_rem(c, _idx(cprA)), _idx(8))
                src = ctx.apply(A_layout, _add(row0, r), _add(c8, k0))
                dst = _add(offA, copy_dst(tiles[0], cprA, it))
                tokens.append(nvgpu_d.device_async_copy(sA, [dst], A, [src], 8, results=[token_ty]))
            for it in range(chunks_B // threads):
                c = _add(tid, _idx(it * threads))
                r, c8 = _div(c, _idx(cprB)), _mul(_rem(c, _idx(cprB)), _idx(8))
                src = ctx.apply(B_layout, _add(r, k0), _add(col0, c8))
                dst = _add(offB, copy_dst(tiles[1], cprB, it))
                tokens.append(nvgpu_d.device_async_copy(sB, [dst], B, [src], 8, results=[token_ty]))
            return tokens

        def load_frags(stage, kk, tiles=(tA, tB)):
            offA, offB = _mul(stage, _idx(fpA)), _mul(stage, _idx(fpB))
            a_frags = [nvgpu_d.ldmatrix(vA, sA, [_add(offA, tiles[0].at(mt * 16, kk * 16))], False, 4) for mt in range(MT)]
            b_frags = []
            for nt2 in range(NT // 2):   # x4.trans: two 8-wide n-tiles per instruction
                b4 = nvgpu_d.ldmatrix(vA, sB, [_add(offB, tiles[1].at(kk * 16, nt2 * 16))], True, 4)
                b_frags.append(vector_d.extract_strided_slice(vB, b4, [0, 0], [2, 2], [1, 1]))
                b_frags.append(vector_d.extract_strided_slice(vB, b4, [2, 0], [2, 2], [1, 1]))
            return a_frags, b_frags

        def compute(stage, acc, tiles=(tA, tB)):
            """ldmatrix + mma over the BK slice held in shared stage `stage`; returns new acc.

            With `prefetch_frags`, all fragments of the slice are loaded before any mma is
            issued (register double-buffering across kk), giving ptxas room to overlap
            ldmatrix latency with tensor-core issue.
            """
            acc = [row[:] for row in acc]
            frag_sets = [load_frags(stage, kk, tiles) for kk in range(BK // 16)] if prefetch_frags else None
            for kk in range(BK // 16):
                a_frags, b_frags = frag_sets[kk] if prefetch_frags else load_frags(stage, kk, tiles)
                for mt in range(MT):
                    for nt in range(NT):
                        acc[mt][nt] = nvgpu_d.mma_sync(vC, a_frags[mt], b_frags[nt], acc[mt][nt], [16, 8, 16])
            return acc

        def commit(tokens):
            grp = nvgpu_d.device_async_create_group(tokens, results=[token_ty])
            return grp

        if S == 1:
            for kt in range(KT):
                grp = commit(issue_loads(_idx(kt), _idx(0)))
                nvgpu_d.device_async_wait(grp, num_groups=0)
                ctx.barrier()
                acc = compute(_idx(0), acc)
                ctx.barrier()
        else:
            # Prologue: stages 0..S-2 in flight, one commit group each.
            groups = [commit(issue_loads(_idx(st), _idx(st))) for st in range(S - 1)]
            flat = [acc[mt][nt] for mt in range(MT) for nt in range(NT)]

            def sub_iteration(kt_base, st, cur, frag_tiles=(tA, tB), copy_tiles=(cA, cB)):
                """K tile kt_base + st lives in stage st (a compile-time constant)."""
                nvgpu_d.device_async_wait(groups[-1], num_groups=S - 2)
                ctx.barrier()
                nxt = _add(kt_base, _idx(st + S - 1))
                clamped = arith_d.MinUIOp(nxt, _idx(KT - 1)).result
                commit(issue_loads(clamped, _idx((st + S - 1) % S), copy_tiles))
                return compute(_idx(st), cur, frag_tiles)

            def loop_body(iv, *carried):
                # Unrolled by S: sub-iteration s handles k-tile iv*S + s in stage s, so every
                # stage index (and hence every shared-memory address) is a compile-time constant.
                cur = [[carried[mt * NT + nt] for nt in range(NT)] for mt in range(MT)]
                kt_base = _mul(iv, _idx(S))
                frag_tiles, copy_tiles = (tA.in_loop(), tB.in_loop()), (cA.in_loop(), cB.in_loop())
                for st in range(S):
                    cur = sub_iteration(kt_base, st, cur, frag_tiles, copy_tiles)
                return [cur[mt][nt] for mt in range(MT) for nt in range(NT)]

            main_iters, rem = divmod(KT, S)
            if main_iters:
                results = ctx.for_range(main_iters, loop_body, init_vals=flat)
                acc = [[results[mt * NT + nt] for nt in range(NT)] for mt in range(MT)]
            # Peeled remainder: the main loop ends on a stage boundary, so the tail restarts at stage 0.
            for st in range(rem):
                acc = sub_iteration(_idx(main_iters * S), st, acc)

        for mt in range(MT):
            for nt in range(NT):
                for i in range(2):
                    r = _add(_add(row0, _add(_mul(wm, _idx(WM)), _idx(mt * 16 + 8 * i))), g_row)
                    pair = [vector_d.extract(acc[mt][nt], [], [i, j]) for j in range(2)]
                    if packed_out:
                        cc2 = _add(_add(_div(col0, _idx(2)), _add(_mul(wn, _idx(WN // 2)), _idx(nt * 4))), _rem(lane, _idx(4)))
                        memref_d.StoreOp(pack_pair(pair, out_dtype), C, [ctx.apply(C_layout, r, cc2)])
                    else:
                        for j in range(2):
                            v = pair[j] if out_dtype == "f32" else arith_d.TruncFOp(f16, pair[j]).result
                            cc = _add(_add(col0, _add(_mul(wn, _idx(WN)), _idx(nt * 8 + j))), g_col2)
                            memref_d.StoreOp(v, C, [ctx.apply(C_layout, r, cc)])

    smem_bytes = S * (fpA + fpB) * 2
    if dynamic_shared is None:
        dynamic_shared = smem_bytes > 48 * 1024
    return KernelBuilder(buffers=bufs, kernel_body=body, name=name,
                         grid=(N // BN, M // BM, 1), block=(threads, 1, 1), dynamic_shared=dynamic_shared)


def pack_pair(pair, out_dtype: str):
    """Two adjacent accumulator values -> one word: f16 pair -> i32, f32 pair -> i64."""
    f16, f32 = F16Type.get(), F32Type.get()
    if out_dtype == "f16":
        elem, word = f16, IntegerType.get_signless(32)
        pair = [arith_d.TruncFOp(f16, v).result for v in pair]
    else:
        elem, word = f32, IntegerType.get_signless(64)
    v2 = VectorType.get([2], elem)
    vec = arith_d.ConstantOp(v2, DenseElementsAttr.get_splat(v2, FloatAttr.get(elem, 0.0))).result
    for j, e in enumerate(pair):
        vec = vector_d.insert(e, vec, [], [j])
    return vector_d.extract(vector_d.bitcast(VectorType.get([1], word), vec), [], [0])


def sA_layout_numel(layout, rows, cols) -> int:
    """Footprint of a tile layout in elements (padded layouts need more than rows*cols)."""
    fp = getattr(layout, "footprint", None)
    return int(fp) if fp is not None else rows * cols
