"""
LEGO WebAssembly compilation backend.

Compiles layout apply/inverse functions to standalone .wasm modules.
Uses the lego-to-wasm MLIR pipeline, then links with wasm-ld.

No GPU hardware required -- this is a cross-compiler.
"""

import base64
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import sympy as sp

from mlir.ir import (
    Context, Location, Module, InsertionPoint,
    IndexType, IntegerType, FunctionType, StringAttr, UnitAttr,
)
from mlir.dialects import func as func_dialect
from mlir.dialects import scf as scf_dialect
from mlir.dialects import arith as arith_dialect
from mlir.passmanager import PassManager

from lego.backend.dialects.lego_dialect import register as register_lego
from lego.backend._ops import (
    _index_const, _emit_row, _emit_col, _emit_reg_p,
    _emit_order_by, _emit_group_by, _emit_tile_by,
    _emit_apply, _emit_apply_inverse,
)
from lego.backend.symbolic import emit_layout_from_python, _resolve_dim


def _find_wasm_ld():
    """Locate wasm-ld linker."""
    path = shutil.which("wasm-ld")
    if path:
        return path
    # Try versioned variants
    for ver in range(20, 14, -1):
        path = shutil.which(f"wasm-ld-{ver}")
        if path:
            return path
    return None


def _build_scalar_apply_module(layout, shape):
    """Build an MLIR module with a scalar @apply function.

    Generates: @apply(i: i32, j: i32, ...) -> i32
    The layout dimensions are baked in as constants.
    """
    rank = len(shape)
    ctx = Context()
    register_lego(ctx)

    with ctx, Location.unknown():
        module = Module.create()
        idx_ty = IndexType.get()
        i32_ty = IntegerType.get_signless(32)

        # --- @apply function: (i32, i32, ...) -> i32 ---
        param_types = [i32_ty] * rank
        func_ty = FunctionType.get(param_types, [i32_ty])

        with InsertionPoint(module.body):
            f = func_dialect.FuncOp("apply", func_ty)
            f.sym_visibility = StringAttr.get("public")
            f.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        entry = f.add_entry_block()
        with InsertionPoint(entry):
            # Cast i32 args to index
            idx_args = []
            for i in range(rank):
                cast = arith_dialect.index_cast(idx_ty, entry.arguments[i])
                idx_args.append(cast)

            # Build layout value with concrete dimension constants
            sym_to_val = {}
            layout_dims = layout._dims if hasattr(layout, '_dims') else layout.dims()
            for d in layout_dims:
                if isinstance(d, sp.Symbol):
                    # Should not happen for concrete shapes, but handle gracefully
                    sym_to_val[d] = _index_const(1)
                elif isinstance(d, sp.Expr):
                    for sym in d.free_symbols:
                        if sym not in sym_to_val:
                            sym_to_val[sym] = _index_const(1)

            # Substitute concrete shape values into symbolic dims
            dim_subs = {}
            actual_dims = layout._dims if hasattr(layout, '_dims') else layout.dims()
            for i_d, d in enumerate(actual_dims):
                if isinstance(d, sp.Expr):
                    for sym in d.free_symbols:
                        # Try to resolve symbol from shape
                        dim_subs[sym] = shape[i_d] if i_d < len(shape) else 1

            # Build sym_to_val with resolved concrete values
            sym_to_val = {}
            all_syms = set()
            for d in actual_dims:
                if isinstance(d, sp.Expr):
                    all_syms |= d.free_symbols
            if hasattr(layout, 'objects'):
                for obj in layout.objects:
                    for d in obj.dims():
                        if isinstance(d, sp.Expr):
                            all_syms |= d.free_symbols
            if hasattr(layout, '_input_chain'):
                for ob in layout._input_chain:
                    for p in ob.perms:
                        for d in p.dims():
                            if isinstance(d, sp.Expr):
                                all_syms |= d.free_symbols
            if hasattr(layout, '_tile_groups'):
                for g in layout._tile_groups:
                    for d in g:
                        if isinstance(d, sp.Expr):
                            all_syms |= d.free_symbols

            # Auto-resolve symbols from positional shape matching
            positional_dims = list(actual_dims)
            for i_d, d in enumerate(positional_dims):
                if isinstance(d, sp.Symbol) and i_d < len(shape):
                    dim_subs[d] = shape[i_d]

            for sym in all_syms:
                val = dim_subs.get(sym, 1)
                sym_to_val[sym] = _index_const(int(val))

            layout_val = emit_layout_from_python(layout, sym_to_val)

            # Apply layout to get flat index
            flat_idx = _emit_apply(layout_val, idx_args)

            # Cast back to i32
            result = arith_dialect.index_cast(i32_ty, flat_idx)
            func_dialect.ReturnOp([result])

        # --- @compute_mapping function: (ptr, i32, i32) -> void ---
        # Writes apply(i, j) for all (i,j) into a flat i32 buffer.
        # Uses LLVM pointer type for the output buffer.
        # We skip this for now — the browser can loop over apply() calls.

    return ctx, module


def compile_to_wasm(layout, shape, opt_level=2):
    """Compile a layout's apply function to a standalone .wasm module.

    Args:
        layout: A LEGO layout object (GroupBy or TileByLayout).
        shape: Concrete integer shape tuple, e.g. (8, 8).
        opt_level: Optimization level (0-3).

    Returns:
        bytes: The standalone .wasm module binary.

    Raises:
        RuntimeError: If compilation or linking fails.
    """
    ctx, module = _build_scalar_apply_module(layout, shape)

    with ctx:
        pipeline = f"builtin.module(lego-to-wasm{{opt-level={opt_level}}})"
        pm = PassManager.parse(pipeline)
        try:
            pm.run(module.operation)
        except Exception as e:
            raise RuntimeError(f"lego-to-wasm pipeline failed:\n{e}") from e

    # Extract base64-encoded binary
    try:
        attr = module.operation.attributes["lego.wasm_binary"]
        wasm_b64 = str(attr).strip('"')
    except KeyError:
        raise RuntimeError(
            "lego-to-wasm pipeline produced no lego.wasm_binary attribute.\n"
            f"Output IR:\n{str(module)[:2000]}"
        )

    wasm_obj = base64.b64decode(wasm_b64)

    # Link with wasm-ld to produce standalone module
    wasm_module = _link_wasm(wasm_obj, exports=["apply"])
    return wasm_module


def _link_wasm(obj_bytes, exports):
    """Link a wasm32 relocatable object into a standalone .wasm module."""
    wasm_ld = _find_wasm_ld()
    if not wasm_ld:
        raise RuntimeError(
            "wasm-ld not found. Install LLVM/LLD or set PATH to include wasm-ld."
        )

    with tempfile.TemporaryDirectory(prefix="lego_wasm_") as tmpdir:
        obj_path = Path(tmpdir) / "layout.o"
        out_path = Path(tmpdir) / "layout.wasm"

        obj_path.write_bytes(obj_bytes)

        cmd = [
            wasm_ld,
            "--no-entry",
            "--export-dynamic",
            "--allow-undefined",  # for any unresolved builtins
            "-o", str(out_path),
            str(obj_path),
        ]
        for name in exports:
            cmd += ["--export", name]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"wasm-ld linking failed:\n{result.stderr}\n{result.stdout}"
            )

        return out_path.read_bytes()


def compile_mapping_json(layout, shape):
    """Compile layout and return mapping as JSON-serializable dict.

    Fallback for when WASM is not needed (server-side evaluation).
    Uses the existing MLIR JIT compiler to compute the permutation table.

    Returns:
        dict with keys: mapping (list of [i, j, flat]), shape, total
    """
    from lego.backend.compiler import LayoutCompiler

    compiler = LayoutCompiler(layout, shape, "i64")
    fwd, inv = compiler.get_permutation_table()

    mapping = []
    total = 1
    for s in shape:
        total *= int(s)

    if len(shape) == 2:
        M, N = int(shape[0]), int(shape[1])
        for i in range(M):
            for j in range(N):
                flat = int(fwd[i * N + j])
                mapping.append([i, j, flat])
    else:
        # General N-D case
        import itertools
        dims = [int(s) for s in shape]
        for coords in itertools.product(*[range(d) for d in dims]):
            linear = sum(c * stride for c, stride in zip(coords, _compute_strides(dims)))
            flat = int(fwd[linear])
            mapping.append(list(coords) + [flat])

    return {"mapping": mapping, "shape": [int(s) for s in shape], "total": total}


def _compute_strides(dims):
    """Compute row-major strides for a shape."""
    strides = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        strides[i] = strides[i + 1] * dims[i + 1]
    return strides
