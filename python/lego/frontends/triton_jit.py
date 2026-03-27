import ast
import atexit
import os
import sys
from dataclasses import dataclass
from functools import reduce

import sympy as sp

from lego.python_printer import LEGOPythonCodePrinter
from lego.frontends._adapter import DSLAdapter
from lego.rewriter import rewrite


# ---------------------------------------------------------------------------
# Block-ptr metadata (Triton-specific)
# ---------------------------------------------------------------------------

@dataclass
class BlockPtrInfo:
    """Structured metadata for generating ``tl.make_block_ptr()`` calls."""
    shape: tuple         # global tensor shape, e.g. (M, K)
    strides: tuple       # memory strides in elements, e.g. (K, 1) for Row
    offsets: tuple        # tile offsets, e.g. (pid_m * BM, k * BK)
    block_shape: tuple    # tile dimensions, e.g. (BM, BK)
    order: tuple          # memory layout order, e.g. (1, 0) for Row
    boundary_dims: tuple  # dimensions needing boundary_check


def extract_block_ptr_metadata(layout, subscript_indices):
    """Extract ``BlockPtrInfo`` from a ``TileByLayout``.

    Only supports single-chain, single-perm (Row or Col) TileBy layouts
    with exactly 2 tile groups (grid + block).  Returns ``None`` otherwise.
    """
    from lego.core import TileByLayout, Row, Col

    if not isinstance(layout, TileByLayout):
        return None

    # Must have single OrderBy in the chain with a single Row/Col perm
    if len(layout._input_chain) != 1:
        return None
    orderby = layout._input_chain[0]
    if len(orderby.perms) != 1:
        return None
    perm = orderby.perms[0]
    if not isinstance(perm, (Row, Col)):
        return None

    # Must have exactly 2 tile groups (grid dims + block dims)
    if len(layout._tile_groups) != 2:
        return None

    shape = tuple(perm.dims())
    ndim = len(shape)
    grid_shape = tuple(layout._tile_groups[0])
    block_shape = tuple(layout._tile_groups[1])

    if len(block_shape) != ndim or len(grid_shape) != ndim:
        return None

    # Compute strides and order based on permutation type
    if isinstance(perm, Row):
        strides = tuple(
            reduce(lambda a, b: a * b, shape[i + 1:], sp.S.One)
            for i in range(ndim)
        )
        order = tuple(range(ndim - 1, -1, -1))
    else:  # Col
        strides = tuple(
            reduce(lambda a, b: a * b, shape[:i], sp.S.One)
            for i in range(ndim)
        )
        order = tuple(range(ndim))

    # Parse subscript: expect [tile_idx, ..., :, :, ...]
    if not isinstance(subscript_indices, (list, tuple)):
        subscript_indices = [subscript_indices]

    tile_indices = []
    slice_count = 0
    for item in subscript_indices:
        if isinstance(item, slice):
            slice_count += 1
        else:
            tile_indices.append(item)

    if len(tile_indices) != ndim or slice_count != ndim:
        return None

    offsets = tuple(tile_indices[i] * block_shape[i] for i in range(ndim))

    # Boundary dims: where grid * block != shape
    boundary_dims = []
    for i in range(ndim):
        diff = sp.simplify(grid_shape[i] * block_shape[i] - shape[i])
        if diff != 0:
            boundary_dims.append(i)

    return BlockPtrInfo(
        shape=shape, strides=strides, offsets=offsets,
        block_shape=block_shape, order=order,
        boundary_dims=tuple(boundary_dims),
    )


# ---------------------------------------------------------------------------
# Triton-specific code printer
# ---------------------------------------------------------------------------

class TritonCodePrinter(LEGOPythonCodePrinter):
    """Renders ``lego_arange`` as ``tl.arange`` for Triton."""

    def _print_lego_arange(self, expr):
        return f"tl.arange({self._print(expr.args[0])}, {self._print(expr.args[1])})"


# ---------------------------------------------------------------------------
# Triton adapter
# ---------------------------------------------------------------------------

class TritonAdapter(DSLAdapter):
    # Triton builtins that take a pointer as first arg
    _POINTER_FUNCS = frozenset((
        'load', 'store', 'atomic_add', 'atomic_max', 'atomic_min',
        'atomic_and', 'atomic_or', 'atomic_xor', 'atomic_xchg', 'atomic_cas',
    ))

    def __init__(self, use_block_ptr=False):
        self.use_block_ptr = use_block_ptr

    def get_rewriter_options(self):
        return {'use_block_ptr': self.use_block_ptr}

    def unwrap(self, fn):
        original_fn = fn
        wrappers = []
        while hasattr(original_fn, 'fn'):
            wrappers.append(original_fn)
            original_fn = original_fn.fn

        source_fn = original_fn
        while hasattr(source_fn, 'src_fn'):
            source_fn = source_fn.src_fn

        return source_fn, original_fn, wrappers

    def find_runtime_vars(self, func_def):
        pointers = set()
        for n in ast.walk(func_def):
            if isinstance(n, ast.Call):
                func_name = ""
                if isinstance(n.func, ast.Attribute):
                    if isinstance(n.func.value, ast.Name) and n.func.value.id == 'tl':
                        func_name = n.func.attr
                elif isinstance(n.func, ast.Name):
                    func_name = n.func.id

                if func_name in self._POINTER_FUNCS:
                    if n.args and isinstance(n.args[0], ast.Name):
                        pointers.add(n.args[0].id)
        return pointers

    def get_code_printer(self):
        return TritonCodePrinter()

    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        _save = os.environ.get('LEGO_SAVE_KERNEL', False)
        temp_dir = os.environ.get("LEGO_TEMP_DIR", "/tmp/lego_kernels")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(
            temp_dir, f"{original_fn.__name__}_{id(original_fn)}.py")

        # Write to file so Triton can use inspect.getsource()
        with open(temp_file, 'w') as f:
            f.write(new_source)

        if return_source:
            if not _save:
                os.remove(temp_file)
            return new_source

        # Compile and execute
        code_obj = compile(tree, filename=temp_file, mode='exec')
        namespace = original_fn.__globals__.copy()
        exec(code_obj, namespace)

        # Register cleanup at exit unless LEGO_SAVE_KERNEL is set
        if not _save:
            atexit.register(
                lambda f=temp_file: os.remove(f) if os.path.exists(f) else None)

        transformed_fn = namespace[original_fn.__name__]
        transformed_fn.__code__ = transformed_fn.__code__.replace(
            co_filename=temp_file)

        # Re-apply Triton wrappers in reverse order
        if wrappers:
            import triton
            from triton.runtime.jit import JITFunction
            from triton.runtime.autotuner import Autotuner
            for wrapper in reversed(wrappers):
                if isinstance(wrapper, Autotuner):
                    transformed_fn = triton.autotune(
                        configs=wrapper.configs,
                        key=wrapper.keys,
                    )(transformed_fn)
                elif isinstance(wrapper, JITFunction):
                    transformed_fn = triton.jit(transformed_fn)

        return transformed_fn


# ---------------------------------------------------------------------------
# Public API (unchanged)
# ---------------------------------------------------------------------------

def jit(fn=None, use_block_ptr=False, **kwargs):
    """
    Decorator that transforms LEGO layout expressions in Triton kernels.

    Usage:
        @lego.jit
        @triton.jit
        def kernel(M, N, K, ...):
            ...

        @lego.jit(use_block_ptr=True)
        @triton.jit
        def kernel(M, N, K, ...):
            ...  # generates tl.make_block_ptr / tl.advance
    """
    def decorator(fn):
        return rewrite(fn, TritonAdapter(use_block_ptr=use_block_ptr), **kwargs)

    if fn is not None:
        return decorator(fn)
    return decorator


def get_kernel_source(fn):
    """
    Utility function to retrieve the raw generated Triton source
    code for a @lego.jit decorated function without compiling it.
    """
    return jit(fn, return_source=True)
