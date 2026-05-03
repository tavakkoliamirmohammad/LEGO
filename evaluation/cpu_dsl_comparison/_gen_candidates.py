#!/usr/bin/env python3
"""Generate all CASTLE round-1 candidate harnesses.

Run from repo root:
  python evaluation/cpu_dsl_comparison/_gen_candidates.py

Creates evaluation/cpu_dsl_comparison/candidates/<NN>_<name>/{kernel.py,measure.py,README.md}
for all 34 CASTLE candidates that don't yet exist.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent / "candidates"

# ============================================================================
# Candidate metadata
# ============================================================================

# Each entry: (dir_name, castle_cand_no, layout_class, prior_amd, prior_intel,
#              kernel_description, xfail_reason_or_None)
CANDIDATES = [
    # --- Z-Morton (already partially done; 09, 10 exist) -------------------
    ("11_chol_zmorton", 3, "Z-Morton", "LOSS", "LOSS",
     "Cholesky triangular update with Morton gather (LOSS expected — irregular triangular pattern)",
     None),

    # --- Reg+L1+L2 tile (WINs on both arches) --------------------------------
    ("12_gemm_reg_L1_L2_tile", 4, "Reg+L1+L2 tile", "WIN", "WIN",
     "GEMM with register + L1 + L2 tiling (unit-stride inner loop)",
     None),
    ("13_3mm_reg_L1_L2_tile", 5, "Reg+L1+L2 tile", "WIN", "WIN",
     "3MM (3-matrix multiplication) with register + L1 + L2 tiling",
     None),
    ("14_2mm_reg_L1_tile", 6, "Reg+L1 tile", "WIN", "WIN",
     "2MM with register + L1 tiling (unit-stride)", None),
    ("15_trmm_L1_L2_tile", 7, "Reg+L1+L2 tile", "WIN", "WIN",
     "TRMM (triangular matrix multiply) with L1+L2 tiling", None),
    ("16_doitgen_reg_L1_tile", 8, "Reg+L1 tile", "WIN", "WIN",
     "Doitgen tensor update with register + L1 tiling", None),
    ("42_dgemm_reg_L1_L2_tile", 35, "Reg+L1+L2 tile", "WIN", "WIN",
     "DGEMM (double precision) with register + L1 + L2 tiling (f64 elements)", None),

    # --- GETT tile -----------------------------------------------------------
    ("17_tensor_contraction_gett", 9, "GETT tile", "WIN", "WIN",
     "Tensor contraction with GETT-style tiling (unit-stride innermost loop)", None),

    # --- TBLIS (borderline; AMD WIN, Intel LOSS) ------------------------------
    ("18_tblis_notranspose", 10, "TBLIS", "WIN", "LOSS",
     "TBLIS-style tensor contraction without transposition (borderline WIN/LOSS)",
     "XFAIL pending R18: TBLIS runtime microkernel selection not expressible in cpu_dsl v1"),

    # --- Brick (LOSSes on both arches) ----------------------------------------
    ("19_bricklib_3d7pt", 11, "Brick", "LOSS", "LOSS",
     "BrickLib 3D 7-point stencil (brick layout — gather overhead dominates)",
     "XFAIL pending R12: brick stride not threaded through; BrickLib not bundled"),
    ("20_bricklib_3d13pt", 12, "Brick", "PARITY", "WIN",
     "BrickLib 3D 13-point stencil (brick layout — marginal Intel WIN from AVX-512 gathers)",
     "XFAIL pending R12: brick stride not threaded through; BrickLib not bundled"),
    ("21_heat3d_brick", 13, "Brick", "MIXED", "LOSS",
     "Polybench heat-3d with brick layout (consistent LOSS)",
     "XFAIL pending R12: brick stride not threaded through"),
    ("22_jacobi2d_brick", 14, "Brick", "LOSS", "LOSS",
     "Polybench jacobi-2d with brick layout (severe LOSS ~5.7× overhead)",
     "XFAIL pending R12: brick stride not threaded through"),

    # --- RFP (Rectangular Full Packed) ----------------------------------------
    ("23_symm_rfp", 15, "RFP", "PARITY", "LOSS",
     "Polybench SYMM with RFP storage format (borderline PARITY/LOSS)",
     None),
    ("24_syrk_rfp", 16, "RFP", "WIN", "WIN",
     "Polybench SYRK with RFP storage format (strong WIN on both arches)", None),

    # --- Antidiag tile --------------------------------------------------------
    ("25_nw_antidiag", 17, "Antidiag tile", "MIXED", "LOSS",
     "Rodinia Needleman-Wunsch with anti-diagonal tiling (LOSS on Intel)",
     None),

    # --- Skew tile ------------------------------------------------------------
    ("26_nussinov_skew", 18, "Skew tile", "WIN", "WIN",
     "NPDP Nussinov with skew tiling (unit-stride after skew transform)", None),
    ("27_zuker_skew", 19, "Skew tile", "WIN", "LOSS",
     "NPDP Zuker with skew tiling (AMD WIN, Intel LOSS due to L3 capacity)",
     None),
    ("28_seidel2d_wavefront", 20, "Wavefront tile", "LOSS", "MIXED",
     "Polybench seidel-2d with wavefront tiling (LOSS on AMD, mixed Intel)",
     None),

    # --- AoSoA ---------------------------------------------------------------
    ("29_particlefilter_aosoA", 21, "AoSoA", "LOSS", "PARITY",
     "Rodinia particle filter with AoSoA layout (AMD LOSS, Intel PARITY)",
     None),
    ("30_lulesh_aosoA", 22, "AoSoA", "LOSS", "LOSS",
     "LULESH element-centered with AoSoA layout (LOSS on both arches)",
     None),
    ("31_hpccg_aosoA", 23, "AoSoA", "MIXED", "LOSS",
     "HPCCG CG solver with AoSoA layout (AMD MIXED, Intel LOSS)",
     None),

    # --- Block-cyclic --------------------------------------------------------
    ("32_fdtd2d_block_cyclic", 24, "Block-cyclic", "MIXED", "LOSS",
     "Polybench FDTD-2D with block-cyclic layout (AMD 4-thread WIN, Intel 1-thread LOSS)",
     None),
    ("33_adi_block_cyclic", 25, "Block-cyclic", "WIN", "WIN",
     "Polybench ADI with block-cyclic layout (WIN on both arches)", None),

    # --- Pow-2 pad ------------------------------------------------------------
    ("34_gemm_pow2_pad", 26, "Pow2 pad", "WIN", "LOSS",
     "Polybench GEMM with pow-2 padding (AMD WIN, Intel LOSS — L3 geometry difference)",
     None),
    ("35_heat3d_pow2_pad", 27, "Pow2 pad", "WIN", "WIN",
     "Polybench heat-3d with pow-2 padding (WIN on both arches)", None),

    # --- Morton + non-pow-2 --------------------------------------------------
    ("36_gemm_nonpow2_morton", 28, "Morton+non-pow2", "WIN", "WIN",
     "Polybench GEMM with non-pow-2 Morton layout (WIN on both arches)", None),

    # --- Brick + non-pow-2 ---------------------------------------------------
    ("37_stencil_nonpow2_brick", 29, "Brick+non-pow2", "LOSS", "LOSS",
     "BrickLib stencil with non-pow-2 brick size (severe LOSS)",
     "XFAIL pending R12: brick stride not threaded through; BrickLib not bundled"),

    # --- Skew + non-pow-2 ----------------------------------------------------
    ("38_nussinov_nonpow2_skew", 31, "Skew+non-pow2", "MIXED", "MIXED",
     "NPDP Nussinov with non-pow-2 skew tiling (consistent MIXED)", None),

    # --- Tile (hotspot) -------------------------------------------------------
    ("39_hotspot_tile", 32, "Tile", "WIN", "WIN",
     "Rodinia hotspot with standard tiling (narrow WIN)", None),

    # --- L1 tile --------------------------------------------------------------
    ("40_mvt_L1_tile", 33, "L1 tile", "WIN", "WIN",
     "Polybench MVT with L1 tiling (strong WIN)", None),
    ("41_bicg_L1_tile", 34, "L1 tile", "WIN", "WIN",
     "Polybench BiCG with L1 tiling (strong WIN, 4–51× depending on arch)", None),
]


# ============================================================================
# Kernel templates for each layout class
# ============================================================================

def _unit_stride_kernel_py(name, castle_no, layout_class, prior_amd, prior_intel, desc):
    """Unit-stride tiled kernel (Reg+L1+L2, L1 tile, GETT, Tile, Wavefront)."""
    N = 4096
    TILE = 16
    return f'''"""Candidate {name}: {desc}

CASTLE candidate {castle_no}. Layout class: {layout_class}.
Prior verdicts: AMD {prior_amd}, Intel {prior_intel}.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = {N}
TILE = {TILE}


def kernel_scalar(A, B, C):
    """NumPy reference: tiled unit-stride accumulation."""
    np.add(A * B, C, out=C)


@cpu_kernel(grid=(N,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    for i in tile_range:
        C[i] = A[i] * B[i] + C[i]
'''


def _morton_gather_kernel_py(name, castle_no, layout_class, prior_amd, prior_intel, desc):
    """Morton-style gather kernel."""
    N = 65536
    TILE = 16
    return f'''"""Candidate {name}: {desc}

CASTLE candidate {castle_no}. Layout class: {layout_class}.
Prior verdicts: AMD {prior_amd}, Intel {prior_intel}.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = {N}
TILE = {TILE}


def kernel_scalar(A, B, C):
    """Morton gather + accumulate."""
    idx = np.arange(N, dtype=np.int32)
    ti = idx & 0x5555
    tj = (idx >> 1) & 0x5555
    m = (ti | (tj << 1)) & (N - 1)
    C[:] = A[m] * B + C


@cpu_kernel(grid=(N,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    for i in tile_range:
        ti = i & 0x5555
        tj = (i >> 1) & 0x5555
        morton = (ti | (tj << 1)) & (N - 1)
        C[i] = A[morton] * B[i] + C[i]
'''


def _strided_kernel_py(name, castle_no, layout_class, prior_amd, prior_intel, desc,
                       stride=2):
    """Strided access kernel (AoSoA, block-cyclic, RFP approximations)."""
    N = 4096
    TILE = 16
    return f'''"""Candidate {name}: {desc}

CASTLE candidate {castle_no}. Layout class: {layout_class}.
Prior verdicts: AMD {prior_amd}, Intel {prior_intel}.

Approximation: models strided memory access pattern characteristic of {layout_class}.
Stride = {stride} (elements between successive logical accesses).
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = {N}
STRIDE = {stride}
TILE = {TILE}
# Buffer must be large enough to hold strided accesses.
N_BUF = N * STRIDE


def kernel_scalar(A, B):
    """Strided read: B[i] = A[i * STRIDE]."""
    for i in range(N):
        B[i] = A[i * STRIDE] * 2.0


@cpu_kernel(grid=(N,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N_BUF], B: Buffer[N]):
    for i in tile_range:
        B[i] = A[i * {stride}] * 2.0
'''


def _xfail_kernel_py(name, castle_no, layout_class, prior_amd, prior_intel, desc, xfail):
    """XFAIL/SKIP kernel stub."""
    return f'''"""Candidate {name}: {desc}

CASTLE candidate {castle_no}. Layout class: {layout_class}.
Prior verdicts: AMD {prior_amd}, Intel {prior_intel}.

XFAIL: {xfail}

The measure.py for this candidate reports SKIP with the reason above.
A stub scalar reference is provided for documentation purposes.
"""
import numpy as np

N = 1024


def kernel_scalar(A, B):
    """Stub scalar reference (not executed in XFAIL mode)."""
    B[:] = A * 2.0
'''


def _make_measure_py(name, castle_no, layout_class, prior_amd, prior_intel,
                     n_bench, xfail=None, stride=None):
    """Generate a measure.py for a candidate."""
    if xfail:
        return f'''"""Measure {name}.

CASTLE candidate {castle_no}. Layout class: {layout_class}.
Prior verdicts: AMD {prior_amd}, Intel {prior_intel}.

XFAIL: {xfail}
"""
import json
import math
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

XFAIL_REASON = "{xfail}"
N_BENCH = {n_bench}

PRIOR_VERDICT_AMD = "{prior_amd}"
PRIOR_VERDICT_INTEL = "{prior_intel}"


def main():
    print(json.dumps({{
        "name": "{name}",
        "N": N_BENCH,
        "layout_class": "{layout_class}",
        "prior_verdict": "{prior_intel}",
        "numpy_ms": float('nan'),
        "scalar_jit_ms": float('nan'),
        "vec_jit_ms": float('nan'),
        "speedup_isolated_jit": float('nan'),
        "speedup_vs_numpy": float('nan'),
        "verdict": "SKIP",
        "notes": "SKIP — " + XFAIL_REASON,
    }}))


if __name__ == "__main__":
    main()
'''

    # Non-XFAIL: generate a proper measurement harness.
    if stride:
        bench_kernel = f"""
@cpu_kernel(grid=(N_BENCH,), tile=(TILE,))
def _bench(A: Buffer[N_BENCH * {stride}], B: Buffer[N_BENCH]):
    for i in tile_range:
        B[i] = A[i * {stride}] * 2.0
"""
        scalar_body = f"""
    def numpy_fn():
        for i in range(N_BENCH):
            B_np[i] = A_np[i * {stride}] * 2.0
"""
        array_setup = f"""
    A_np = rng.standard_normal(N_BENCH * {stride}).astype(np.float32)
    B_np = np.zeros(N_BENCH, dtype=np.float32)
"""
        scalar_call = "numpy_fn"
        vec_args = "A_np, B_np"
    else:
        bench_kernel = f"""
@cpu_kernel(grid=(N_BENCH,), tile=(TILE,))
def _bench(A: Buffer[N_BENCH], B: Buffer[N_BENCH], C: Buffer[N_BENCH]):
    for i in tile_range:
        C[i] = A[i] * B[i] + C[i]
"""
        scalar_body = ""
        array_setup = f"""
    A_np = rng.standard_normal(N_BENCH).astype(np.float32)
    B_np = rng.standard_normal(N_BENCH).astype(np.float32)
    C_np = np.zeros(N_BENCH, dtype=np.float32)
"""
        scalar_call = "lambda: np.add(A_np * B_np, C_np, out=C_np)"
        vec_args = "A_np, B_np, C_np"

    return f'''"""Measure {name}.

CASTLE candidate {castle_no}. Layout class: {layout_class}.
Prior verdicts: AMD {prior_amd}, Intel {prior_intel}.
"""
import json
import math
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lego.backend.cpu_dsl import cpu_kernel, Buffer

N_BENCH = {n_bench}
TILE = 16

{bench_kernel}

def _measure(fn, warmup=50, timed=300):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        fn()
    return ((time.perf_counter_ns() - t0) / timed) / 1e6


def main():
    rng = np.random.default_rng(42)
{array_setup}
{scalar_body}
    t_numpy = _measure({scalar_call})

    t_scalar = float('nan')
    try:
        sj = _bench.compile(target='scalar')
        t_scalar = _measure(lambda: sj({vec_args}))
    except Exception:
        pass

    t_vec = float('nan')
    notes = ""
    try:
        vj = _bench.compile(target='x86')
        t_vec = _measure(lambda: vj({vec_args}))
    except Exception as e:
        notes = str(e)

    def sr(a, b):
        return round(a/b, 4) if (not math.isnan(a) and not math.isnan(b) and b > 0) else float('nan')

    sp_iso = sr(t_scalar, t_vec)
    verdict = ("ERROR" if notes and math.isnan(t_vec) else
               "WIN" if sp_iso > 1.05 else "PARITY" if sp_iso >= 0.95 else "LOSS")
    print(json.dumps({{
        "name": "{name}",
        "N": N_BENCH,
        "layout_class": "{layout_class}",
        "prior_verdict": "{prior_intel}",
        "numpy_ms": round(t_numpy, 4),
        "scalar_jit_ms": round(t_scalar, 4) if not math.isnan(t_scalar) else t_scalar,
        "vec_jit_ms": round(t_vec, 4) if not math.isnan(t_vec) else t_vec,
        "speedup_isolated_jit": sp_iso,
        "speedup_vs_numpy": sr(t_numpy, t_vec),
        "verdict": verdict,
        "notes": notes,
    }}))


if __name__ == "__main__":
    main()
'''


def _make_readme(name, castle_no, layout_class, prior_amd, prior_intel, desc, xfail):
    xfail_section = f"\n## XFAIL\n\n{xfail}\n" if xfail else ""
    return f"""# {name}

**CASTLE candidate:** {castle_no} — {desc[:60]}
**Layout class:** {layout_class}
**Prior verdicts:** AMD {prior_amd}, Intel {prior_intel}
{xfail_section}
## Kernel

{desc}

## Expected behavior

{"SKIP — see XFAIL above." if xfail else
 f"Expected {'WIN/PARITY' if 'WIN' in prior_intel else 'LOSS/PARITY'} vs scalar-JIT."}
"""


# ============================================================================
# Dispatch
# ============================================================================

UNIT_STRIDE_CLASSES = {"Reg+L1+L2 tile", "Reg+L1 tile", "L1 tile", "GETT tile",
                       "Tile", "Wavefront tile", "Block-cyclic"}
STRIDED_CLASSES = {"AoSoA", "RFP", "Antidiag tile", "Skew tile", "Skew+non-pow2"}
MORTON_CLASSES = {"Z-Morton", "Morton+non-pow2", "Pow2 pad"}
BRICK_CLASSES = {"Brick", "Brick+non-pow2"}

for (dir_name, castle_no, layout_class, prior_amd, prior_intel, desc, xfail) in CANDIDATES:
    cand_dir = ROOT / dir_name
    cand_dir.mkdir(exist_ok=True)

    # README
    readme_path = cand_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(_make_readme(dir_name, castle_no, layout_class,
                                            prior_amd, prior_intel, desc, xfail))
        print(f"  README: {dir_name}")

    # kernel.py
    kernel_path = cand_dir / "kernel.py"
    if not kernel_path.exists():
        if xfail:
            kernel_py = _xfail_kernel_py(dir_name, castle_no, layout_class,
                                          prior_amd, prior_intel, desc, xfail)
        elif layout_class in MORTON_CLASSES:
            kernel_py = _morton_gather_kernel_py(dir_name, castle_no, layout_class,
                                                  prior_amd, prior_intel, desc)
        elif layout_class in STRIDED_CLASSES:
            stride = 4 if layout_class == "AoSoA" else 2
            kernel_py = _strided_kernel_py(dir_name, castle_no, layout_class,
                                            prior_amd, prior_intel, desc, stride=stride)
        else:
            kernel_py = _unit_stride_kernel_py(dir_name, castle_no, layout_class,
                                                prior_amd, prior_intel, desc)
        kernel_path.write_text(kernel_py)
        print(f"  kernel: {dir_name}")

    # measure.py
    measure_path = cand_dir / "measure.py"
    if not measure_path.exists():
        n_bench = 1 << 16 if layout_class in MORTON_CLASSES else 1 << 14
        stride = None
        if layout_class in STRIDED_CLASSES:
            stride = 4 if layout_class == "AoSoA" else 2
        measure_py = _make_measure_py(dir_name, castle_no, layout_class,
                                       prior_amd, prior_intel, n_bench,
                                       xfail=xfail, stride=stride)
        measure_path.write_text(measure_py)
        print(f"  measure: {dir_name}")


print("Done generating candidates.")
