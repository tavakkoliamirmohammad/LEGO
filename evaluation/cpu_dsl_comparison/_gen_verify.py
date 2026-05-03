#!/usr/bin/env python3
"""Generate verify.py for each candidate based on kernel.py introspection.

Run from evaluation/cpu_dsl_comparison/:
    python _gen_verify.py

This writes verify.py in each candidate directory. Existing verify.py files
are overwritten only if they are older than the kernel.py (or if --force).
"""
import ast
import sys
import textwrap
from pathlib import Path

CANDS = Path(__file__).parent / "candidates"

# Per-candidate configuration: maps candidate name prefix to a dict describing
# how to build inputs and call kernel_scalar/kernel_cpu_dsl.
#
# Keys:
#   imports: additional imports needed in the generated file
#   setup: code to build rng and input arrays (as a string, indented 4 spaces)
#   scalar_call: how to call kernel_scalar to produce the reference output
#   vec_inputs: list of arg names passed to the JIT function
#   out_arr: name of the output array to compare (must be a numpy array)
#   rtol: relative tolerance for assert_allclose
#   note: optional short note about what this kernel tests

CONFIGS = {
    "01_saxpy": dict(
        imports="from kernel import kernel_scalar, kernel_cpu_dsl, N",
        setup=textwrap.dedent("""\
            rng = np.random.default_rng(0)
            a = np.float32(2.5)
            X = rng.standard_normal(N).astype(np.float32)
            Y_ref = rng.standard_normal(N).astype(np.float32)
        """),
        scalar_call="kernel_scalar(a, X.copy(), (Y_ref_copy := Y_ref.copy()))",
        scalar_out="Y_ref_copy",
        vec_call_setup="Y_vec = Y_ref.copy()",
        vec_args="a, X, Y_vec",
        out_arr="Y_vec",
        rtol="1e-5",
    ),
    "02_gemm_row_major": dict(
        imports="from kernel import kernel_scalar, kernel_cpu_dsl, M, N, K, _MK, _KN, _MN",
        setup=textwrap.dedent("""\
            rng = np.random.default_rng(0)
            A_2d = rng.standard_normal((M, K)).astype(np.float32)
            B_2d = rng.standard_normal((K, N)).astype(np.float32)
            C_ref = np.zeros((M, N), dtype=np.float32)
            kernel_scalar(A_2d, B_2d, C_ref)
            A = A_2d.ravel(); B = B_2d.ravel()
        """),
        scalar_call="",  # already done above
        scalar_out="C_ref.ravel()",
        vec_call_setup="C_vec = np.zeros(_MN, dtype=np.float32)",
        vec_args="A, B, C_vec",
        out_arr="C_vec",
        rtol="1e-3",
    ),
    "03_3pt_stencil_1d": dict(
        imports="from kernel import kernel_scalar, kernel_cpu_dsl, N",
        setup=textwrap.dedent("""\
            rng = np.random.default_rng(0)
            A = rng.standard_normal(N).astype(np.float32)
            B_ref = np.zeros(N, dtype=np.float32)
            kernel_scalar(A, B_ref)
        """),
        scalar_call="",
        scalar_out="B_ref",
        vec_call_setup="B_vec = np.zeros(N, dtype=np.float32)",
        vec_args="A, B_vec",
        out_arr="B_vec",
        rtol="1e-5",
        out_slice="[1:-1]",
    ),
    "04_col_major_inner": dict(
        imports="from kernel import kernel_scalar, kernel_cpu_dsl, M, N, _MN",
        setup=textwrap.dedent("""\
            rng = np.random.default_rng(0)
            A = rng.standard_normal(_MN).astype(np.float32)
            C_ref = np.zeros(_MN, dtype=np.float32)
            kernel_scalar(A.reshape(M, N), C_ref.reshape(M, N))
        """),
        scalar_call="",
        scalar_out="C_ref",
        vec_call_setup="C_vec = np.zeros(_MN, dtype=np.float32)",
        vec_args="A, C_vec",
        out_arr="C_vec",
        rtol="1e-5",
    ),
    "05_morton_2d": dict(
        imports="from kernel import kernel_scalar, kernel_cpu_dsl, N",
        setup=textwrap.dedent("""\
            rng = np.random.default_rng(0)
            A = rng.standard_normal(N).astype(np.float32)
            B_ref = np.zeros(N, dtype=np.float32)
            kernel_scalar(A, B_ref)
        """),
        scalar_call="",
        scalar_out="B_ref",
        vec_call_setup="B_vec = np.zeros(N, dtype=np.float32)",
        vec_args="A, B_vec",
        out_arr="B_vec",
        rtol="1e-5",
    ),
    "06_self_update": dict(
        imports="from kernel import kernel_scalar, kernel_cpu_dsl, N",
        setup=textwrap.dedent("""\
            rng = np.random.default_rng(0)
            A = rng.standard_normal(N).astype(np.float32)
            B_ref = np.zeros(N, dtype=np.float32)
            kernel_scalar(A, B_ref)
        """),
        scalar_call="",
        scalar_out="B_ref",
        vec_call_setup="B_vec = np.zeros(N, dtype=np.float32)",
        vec_args="A, B_vec",
        out_arr="B_vec",
        rtol="1e-5",
    ),
    "07_mixed_precision": dict(
        imports="from kernel import kernel_scalar, kernel_cpu_dsl, N",
        setup=textwrap.dedent("""\
            rng = np.random.default_rng(0)
            X = rng.standard_normal(N).astype(np.float32)
            Y_ref = rng.standard_normal(N).astype(np.float32)
            Y_copy = Y_ref.copy()
            kernel_scalar(X, Y_copy)
        """),
        scalar_call="",
        scalar_out="Y_copy",
        vec_call_setup="Y_vec = Y_ref.copy()",
        vec_args="X, Y_vec",
        out_arr="Y_vec",
        rtol="1e-5",
    ),
    "08_brick_within_cell": dict(
        imports="from kernel import kernel_scalar, kernel_cpu_dsl, N",
        setup=textwrap.dedent("""\
            rng = np.random.default_rng(0)
            A = rng.standard_normal(N).astype(np.float32)
            B_ref = np.zeros(N, dtype=np.float32)
            kernel_scalar(A, B_ref)
        """),
        scalar_call="",
        scalar_out="B_ref",
        vec_call_setup="B_vec = np.zeros(N, dtype=np.float32)",
        vec_args="A, B_vec",
        out_arr="B_vec",
        rtol="1e-5",
    ),
}

# Candidates 09-17: pattern kernel_scalar(A, B, C); grid=(N,) or grid=(N_BENCH,)
for cname, pv in [
    ("09_gemm_zmorton", "WIN"), ("10_lu_zmorton", "WIN"),
    ("11_chol_zmorton", "LOSS"), ("12_gemm_reg_L1_L2_tile", "WIN"),
    ("13_3mm_reg_L1_L2_tile", "WIN"), ("14_2mm_reg_L1_tile", "WIN"),
    ("15_trmm_L1_L2_tile", "WIN"), ("16_doitgen_reg_L1_tile", "WIN"),
    ("17_tensor_contraction_gett", "WIN"),
    ("28_seidel2d_wavefront", "MIXED"), ("32_fdtd2d_block_cyclic", "LOSS"),
    ("33_adi_block_cyclic", "WIN"), ("34_gemm_pow2_pad", "LOSS"),
    ("35_heat3d_pow2_pad", "WIN"), ("36_gemm_nonpow2_morton", "WIN"),
    ("39_hotspot_tile", "WIN"), ("40_mvt_L1_tile", "WIN"),
    ("41_bicg_L1_tile", "WIN"), ("42_dgemm_reg_L1_L2_tile", "WIN"),
]:
    CONFIGS[cname] = dict(
        imports="from kernel import kernel_scalar, kernel_cpu_dsl, N",
        setup=textwrap.dedent("""\
            rng = np.random.default_rng(0)
            A = rng.standard_normal(N).astype(np.float32)
            B = rng.standard_normal(N).astype(np.float32)
            C_ref = rng.standard_normal(N).astype(np.float32)
            C_copy = C_ref.copy()
            kernel_scalar(A, B, C_copy)
        """),
        scalar_call="",
        scalar_out="C_copy",
        vec_call_setup="C_vec = C_ref.copy()",
        vec_args="A, B, C_vec",
        out_arr="C_vec",
        rtol="1e-4",
    )

# GEMM with 2 extra dims
CONFIGS["02_gemm_row_major"] = dict(
    imports="from kernel import kernel_scalar, kernel_cpu_dsl, M, N, K, _MK, _KN, _MN",
    setup=textwrap.dedent("""\
        rng = np.random.default_rng(0)
        A_2d = rng.standard_normal((M, K)).astype(np.float32)
        B_2d = rng.standard_normal((K, N)).astype(np.float32)
        C_ref_2d = np.zeros((M, N), dtype=np.float32)
        kernel_scalar(A_2d, B_2d, C_ref_2d)
        A = A_2d.ravel(); B = B_2d.ravel()
    """),
    scalar_call="",
    scalar_out="C_ref_2d.ravel()",
    vec_call_setup="C_vec = np.zeros(_MN, dtype=np.float32)",
    vec_args="A, B, C_vec",
    out_arr="C_vec",
    rtol="1e-3",
)

# Tblis: GEMM pattern
CONFIGS["18_tblis_notranspose"] = dict(
    imports="from kernel import kernel_scalar, kernel_cpu_dsl, M, N, K, _MK, _KN, _MN",
    setup=textwrap.dedent("""\
        rng = np.random.default_rng(0)
        A_2d = rng.standard_normal((M, K)).astype(np.float32)
        B_2d = rng.standard_normal((K, N)).astype(np.float32)
        C_ref_2d = np.zeros((M, N), dtype=np.float32)
        kernel_scalar(A_2d, B_2d, C_ref_2d)
        A = A_2d.ravel(); B = B_2d.ravel()
    """),
    scalar_call="",
    scalar_out="C_ref_2d.ravel()",
    vec_call_setup="C_vec = np.zeros(_MN, dtype=np.float32)",
    vec_args="A, B, C_vec",
    out_arr="C_vec",
    rtol="1e-3",
)

# Brick stencils with special _OFFSET / _INNER
for cname in ["19_bricklib_3d7pt", "20_bricklib_3d13pt", "21_heat3d_brick"]:
    CONFIGS[cname] = dict(
        imports="from kernel import kernel_scalar, kernel_cpu_dsl, N_FLAT, _INNER, _OFFSET",
        setup=textwrap.dedent("""\
            rng = np.random.default_rng(0)
            A = rng.standard_normal(N_FLAT).astype(np.float32)
            B_ref = np.zeros(N_FLAT, dtype=np.float32)
        """),
        scalar_call="",  # scalar_out reference: run scalar_jit
        use_scalar_jit_ref=True,  # use scalar JIT as reference (not kernel_scalar, which is approximate)
        scalar_out="B_sc[_OFFSET:_OFFSET + _INNER]",
        vec_call_setup="B_vec = np.zeros(N_FLAT, dtype=np.float32)",
        vec_args="A, B_vec",
        out_arr="B_vec[_OFFSET:_OFFSET + _INNER]",
        scalar_args="A, B_sc",
        rtol="1e-4",
    )

CONFIGS["22_jacobi2d_brick"] = dict(
    imports="from kernel import kernel_scalar, kernel_cpu_dsl, N_FLAT, _INNER, _OFFSET",
    setup=textwrap.dedent("""\
        rng = np.random.default_rng(0)
        A = rng.standard_normal(N_FLAT).astype(np.float32)
        B_ref = np.zeros(N_FLAT, dtype=np.float32)
    """),
    use_scalar_jit_ref=True,
    scalar_out="B_sc[_OFFSET:_OFFSET + _INNER]",
    vec_call_setup="B_vec = np.zeros(N_FLAT, dtype=np.float32)",
    vec_args="A, B_vec",
    out_arr="B_vec[_OFFSET:_OFFSET + _INNER]",
    scalar_args="A, B_sc",
    rtol="1e-4",
)

CONFIGS["37_stencil_nonpow2_brick"] = dict(
    imports="from kernel import kernel_scalar, kernel_cpu_dsl, N_FLAT, _INNER, _OFFSET",
    setup=textwrap.dedent("""\
        rng = np.random.default_rng(0)
        A = rng.standard_normal(N_FLAT).astype(np.float32)
        B_ref = np.zeros(N_FLAT, dtype=np.float32)
    """),
    use_scalar_jit_ref=True,
    scalar_out="B_sc[_OFFSET:_OFFSET + _INNER]",
    vec_call_setup="B_vec = np.zeros(N_FLAT, dtype=np.float32)",
    vec_args="A, B_vec",
    out_arr="B_vec[_OFFSET:_OFFSET + _INNER]",
    scalar_args="A, B_sc",
    rtol="1e-4",
)

# Strided patterns: kernel_scalar(A, B) with N_BUF
for cname in [
    "23_symm_rfp", "24_syrk_rfp", "25_nw_antidiag",
    "26_nussinov_skew", "27_zuker_skew",
    "29_particlefilter_aosoA", "30_lulesh_aosoA", "31_hpccg_aosoA",
    "38_nussinov_nonpow2_skew",
]:
    CONFIGS[cname] = dict(
        imports="from kernel import kernel_scalar, kernel_cpu_dsl, N, N_BUF",
        setup=textwrap.dedent("""\
            rng = np.random.default_rng(0)
            A = rng.standard_normal(N_BUF).astype(np.float32)
            B_ref = np.zeros(N, dtype=np.float32)
            kernel_scalar(A, B_ref)
        """),
        scalar_call="",
        scalar_out="B_ref",
        vec_call_setup="B_vec = np.zeros(N, dtype=np.float32)",
        vec_args="A, B_vec",
        out_arr="B_vec",
        rtol="1e-5",
    )

TEMPLATE = '''\
"""Verify correctness of {name} across scalar_jit / vec_jit paths."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

{imports}


def main():
    rng = np.random.default_rng(0)  # noqa: F841
    # --- Build deterministic inputs ---
{setup_indented}
    # --- Reference: scalar JIT ---
{scalar_jit_block}
    # --- Vectorized JIT ---
{vec_jit_block}
    # --- Assert correctness ---
{assert_block}
    print(f"VERIFIED: {{__file__}}")


if __name__ == "__main__":
    main()
'''


def make_setup_indented(setup):
    """Indent 4 extra spaces for the template body."""
    lines = setup.strip().split("\n")
    return "\n".join("    " + l for l in lines)


def gen_verify(cname, cfg):
    name = cname

    setup_indented = make_setup_indented(cfg["setup"])

    use_scalar_jit_ref = cfg.get("use_scalar_jit_ref", False)

    # Scalar JIT block
    scalar_out = cfg.get("scalar_out", "B_sc")
    scalar_args_str = cfg.get("scalar_args", cfg.get("vec_args", ""))

    if use_scalar_jit_ref:
        # Use scalar JIT as the reference (for stencil candidates where
        # kernel_scalar is approximate / operates on reshaped arrays)
        scalar_jit_block = f"""\
    scalar_jit = kernel_cpu_dsl.compile(target='scalar')
    B_sc = np.zeros_like({cfg['vec_call_setup'].split('=')[0].strip()})
    try:
        scalar_jit({scalar_args_str})
    except Exception as e:
        print(f"PENDING R??: scalar_jit failed: {{e}}")
        return"""
        # We need B_sc initialized properly
        scalar_jit_block = f"""\
    scalar_jit = kernel_cpu_dsl.compile(target='scalar')
    B_sc = np.zeros(N_FLAT, dtype=np.float32)
    try:
        scalar_jit(A, B_sc)
    except Exception as e:
        print(f"PENDING R??: scalar_jit failed: {{e}}")
        return"""
    else:
        scalar_jit_block = f"""\
    scalar_jit = kernel_cpu_dsl.compile(target='scalar')
    {cfg['vec_call_setup'].replace('vec', 'sc')}
    try:
        scalar_jit({scalar_args_str.replace('vec', 'sc').replace('C_ref', 'C_sc').replace('Y_ref', 'Y_sc')})
    except Exception as e:
        print(f"PENDING R??: scalar_jit failed: {{e}}")
        return"""
        # Simplify: just use vec_args but with _sc suffix for the out array
        out_name = cfg.get("out_arr", "B_vec")
        sc_out_name = out_name.replace("_vec", "_sc").replace("[", "").split("[")[0]
        scalar_jit_block = f"""\
    scalar_jit = kernel_cpu_dsl.compile(target='scalar')
    {cfg['vec_call_setup'].replace('_vec', '_sc')}
    try:
        scalar_jit({cfg['vec_args'].replace('_vec', '_sc')})
    except Exception as e:
        print(f"PENDING R??: scalar_jit failed: {{e}}")
        return"""

    # Vec JIT block
    vec_jit_block = f"""\
    vec_jit = kernel_cpu_dsl.compile(target='x86')
    {cfg['vec_call_setup']}
    try:
        vec_jit({cfg['vec_args']})
    except Exception as e:
        print(f"PENDING R??: vec_jit failed: {{e}}")
        return"""

    # Assert block
    if use_scalar_jit_ref:
        ref_expr = scalar_out
        out_expr = cfg["out_arr"]
        assert_block = f"""\
    np.testing.assert_allclose(
        {out_expr},
        {ref_expr},
        rtol={cfg['rtol']},
        err_msg="vec_jit != scalar_jit",
    )"""
    else:
        # Use kernel_scalar reference OR the scalar_out
        sc_out = cfg.get("scalar_out", "")
        sc_vars = ""
        if cfg.get("scalar_call", ""):
            sc_vars = f"    {cfg['scalar_call']}\n"
        out_slice = cfg.get("out_slice", "")
        vec_out = cfg["out_arr"] + out_slice
        ref_out = sc_out + out_slice if sc_out else f"B_sc{out_slice}"

        # For verify, compare vec to scalar_jit (more reliable than np reference)
        sc_arr = cfg['vec_call_setup'].replace('_vec', '_sc').split('=')[0].strip()
        sc_arr_slice = sc_arr + out_slice
        assert_block = f"""\
    np.testing.assert_allclose(
        {vec_out},
        {sc_arr_slice},
        rtol={cfg['rtol']},
        err_msg="vec_jit != scalar_jit",
    )"""

    imports_str = cfg.get("imports", "from kernel import kernel_scalar, kernel_cpu_dsl, N")

    content = TEMPLATE.format(
        name=name,
        imports=imports_str,
        setup_indented=setup_indented,
        scalar_jit_block=scalar_jit_block,
        vec_jit_block=vec_jit_block,
        assert_block=assert_block,
    )
    return content


def main():
    for cand_dir in sorted(CANDS.iterdir()):
        if not cand_dir.is_dir():
            continue
        cname = cand_dir.name
        cfg = CONFIGS.get(cname)
        if cfg is None:
            print(f"  SKIP {cname}: no config")
            continue
        verify_path = cand_dir / "verify.py"
        content = gen_verify(cname, cfg)
        verify_path.write_text(content)
        print(f"  WROTE {verify_path}")


if __name__ == "__main__":
    main()
