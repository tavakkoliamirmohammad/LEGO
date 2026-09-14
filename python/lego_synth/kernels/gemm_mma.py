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
                          IntegerAttr, FloatAttr)
from lego.mlir.dialects import arith as arith_d, memref as memref_d, nvgpu as nvgpu_d, vector as vector_d, gpu as gpu_d


def _idx(v: int):
    return arith_d.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), int(v))).result


def _add(a, b): return arith_d.AddIOp(a, b).result
def _mul(a, b): return arith_d.MulIOp(a, b).result
def _div(a, b): return arith_d.DivUIOp(a, b).result
def _rem(a, b): return arith_d.RemUIOp(a, b).result


def build_gemm(M: int, N: int, K: int, BM: int, BN: int, BK: int, sA_layout, sB_layout,
               warps_m: int = 2, warps_n: int = 2, name: str = "gemm_mma", unroll_k: bool = False,
               stages: int = 3, dynamic_shared: bool = None) -> KernelBuilder:
    """stages > 1 enables a multi-stage cp.async pipeline (unroll_k forces the single-stage form)."""
    assert M % BM == 0 and N % BN == 0 and K % BK == 0 and BK % 16 == 0
    WM, WN = BM // warps_m, BN // warps_n
    assert WM % 16 == 0 and WN % 16 == 0, "B fragments are loaded two n-tiles at a time (ldmatrix.x4.trans)"
    threads = 32 * warps_m * warps_n
    MT, NT = WM // 16, WN // 8
    KT = K // BK
    S = 1 if unroll_k else max(1, stages)
    assert S <= KT
    A_layout, B_layout, C_layout = Row(M, K), Row(K, N), Row(M, N)
    fpA, fpB = sA_layout_numel(sA_layout, BM, BK), sA_layout_numel(sB_layout, BK, BN)

    bufs = [LayoutBuffer(A_layout, shape=(M * K,), dtype=DType.f16),
            LayoutBuffer(B_layout, shape=(K * N,), dtype=DType.f16),
            LayoutBuffer(C_layout, shape=(M * N,), dtype=DType.f32),
            LayoutBuffer(sA_layout, shape=(S * fpA,), dtype=DType.f16, shared=True),
            LayoutBuffer(sB_layout, shape=(S * fpB,), dtype=DType.f16, shared=True)]

    def body(ctx):
        A, B, C, sA, sB = ctx._buf_vals
        f16, f32 = F16Type.get(), F32Type.get()
        vA = VectorType.get([4, 2], f16); vB = VectorType.get([2, 2], f16); vC = VectorType.get([2, 2], f32)
        token_ty = Type.parse("!nvgpu.device.async.token")
        tid = ctx.thread_id.x
        bx, by = ctx.block_id.x, ctx.block_id.y
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

        def issue_loads(kt_idx, stage):
            """cp.async the A and B tiles of k-slice kt_idx into shared stage `stage`; returns tokens."""
            k0 = _mul(kt_idx, _idx(BK))
            offA, offB = _mul(stage, _idx(fpA)), _mul(stage, _idx(fpB))
            tokens = []
            for it in range(chunks_A // threads):
                c = _add(tid, _idx(it * threads))
                r, c8 = _div(c, _idx(BK // 8)), _mul(_rem(c, _idx(BK // 8)), _idx(8))
                src = ctx.apply(A_layout, _add(row0, r), _add(c8, k0))
                dst = _add(offA, ctx.apply(sA_layout, r, c8))
                tokens.append(nvgpu_d.device_async_copy(sA, [dst], A, [src], 8, results=[token_ty]))
            for it in range(chunks_B // threads):
                c = _add(tid, _idx(it * threads))
                r, c8 = _div(c, _idx(BN // 8)), _mul(_rem(c, _idx(BN // 8)), _idx(8))
                src = ctx.apply(B_layout, _add(r, k0), _add(col0, c8))
                dst = _add(offB, ctx.apply(sB_layout, r, c8))
                tokens.append(nvgpu_d.device_async_copy(sB, [dst], B, [src], 8, results=[token_ty]))
            return tokens

        def compute(stage, acc):
            """ldmatrix + mma over the BK slice held in shared stage `stage`; returns new acc."""
            offA, offB = _mul(stage, _idx(fpA)), _mul(stage, _idx(fpB))
            acc = [row[:] for row in acc]
            for kk in range(BK // 16):
                a_frags = []
                for mt in range(MT):
                    r = _add(_add(_mul(wm, _idx(WM)), _idx(mt * 16)), lane16)
                    cidx = _add(_idx(kk * 16), lane_hi)
                    a_frags.append(nvgpu_d.ldmatrix(vA, sA, [_add(offA, ctx.apply(sA_layout, r, cidx))], False, 4))
                b_frags = []
                for nt2 in range(NT // 2):   # x4.trans: two 8-wide n-tiles per instruction
                    r = _add(_idx(kk * 16), lane16)
                    cidx = _add(_add(_mul(wn, _idx(WN)), _idx(nt2 * 16)), lane_hi)
                    b4 = nvgpu_d.ldmatrix(vA, sB, [_add(offB, ctx.apply(sB_layout, r, cidx))], True, 4)
                    b_frags.append(vector_d.extract_strided_slice(vB, b4, [0, 0], [2, 2], [1, 1]))
                    b_frags.append(vector_d.extract_strided_slice(vB, b4, [2, 0], [2, 2], [1, 1]))
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
            assert KT % S == 0, "K tiles must be a multiple of the pipeline depth (loop is unrolled by S)"
            groups = [commit(issue_loads(_idx(st), _idx(st))) for st in range(S - 1)]
            flat = [acc[mt][nt] for mt in range(MT) for nt in range(NT)]

            def loop_body(iv, *carried):
                # Unrolled by S: sub-iteration s handles k-tile iv*S + s in stage s, so every
                # stage index (and hence every shared-memory address) is a compile-time constant.
                cur = [[carried[mt * NT + nt] for nt in range(NT)] for mt in range(MT)]
                kt_base = _mul(iv, _idx(S))
                for st in range(S):
                    nvgpu_d.device_async_wait(groups[-1], num_groups=S - 2)
                    ctx.barrier()
                    nxt = _add(kt_base, _idx(st + S - 1))
                    clamped = arith_d.MinUIOp(nxt, _idx(KT - 1)).result
                    commit(issue_loads(clamped, _idx((st + S - 1) % S)))
                    cur = compute(_idx(st), cur)
                return [cur[mt][nt] for mt in range(MT) for nt in range(NT)]

            results = ctx.for_range(KT // S, loop_body, init_vals=flat)
            acc = [[results[mt * NT + nt] for nt in range(NT)] for mt in range(MT)]

        for mt in range(MT):
            for nt in range(NT):
                for i in range(2):
                    for j in range(2):
                        v = vector_d.extract(acc[mt][nt], [], [i, j])
                        r = _add(_add(row0, _add(_mul(wm, _idx(WM)), _idx(mt * 16 + 8 * i))), g_row)
                        cc = _add(_add(col0, _add(_mul(wn, _idx(WN)), _idx(nt * 8 + j))), g_col2)
                        memref_d.StoreOp(v, C, [ctx.apply(C_layout, r, cc)])

    smem_bytes = S * (fpA + fpB) * 2
    if dynamic_shared is None:
        dynamic_shared = smem_bytes > 48 * 1024
    return KernelBuilder(buffers=bufs, kernel_body=body, name=name,
                         grid=(N // BN, M // BM, 1), block=(threads, 1, 1), dynamic_shared=dynamic_shared)


def sA_layout_numel(layout, rows, cols) -> int:
    """Footprint of a tile layout in elements (padded layouts need more than rows*cols)."""
    fp = getattr(layout, "footprint", None)
    return int(fp) if fp is not None else rows * cols
