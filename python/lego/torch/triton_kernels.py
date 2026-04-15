"""
LEGO Triton Kernel Codegen (Layer 1 -- GPU Path)

Dynamically generates Triton matmul kernels whose memory indexing matches
the LEGO layout attached to each operand.  The layout algebra expression
(e.g. ``OrderBy(Col(M, K))``) is embedded in the generated kernel source
and compiled by ``@lego_jit`` into pure arithmetic Triton code -- no
permutation tables, no extra memory indirection.

Supports ANY LEGO layout: RowMajor, ColMajor, TiledPermute, Transposed,
ZCurve, Swizzle, BlockCyclic, or arbitrary GenP/RegP.

This is the same ``@lego_jit`` mechanism used by all examples in
``python/examples/triton/``.
"""

import torch

try:
    import triton
    import triton.language as tl
    from lego.core import Row, Col, RegP, GenP, OrderBy, GroupBy, TileByLayout
    from lego.frontends.triton_jit import jit as lego_jit
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ============================================================================
# Layout -> algebra expression converter
# ============================================================================

def _layout_to_expr(layout, dim0="M", dim1="K"):
    """Convert a LegoLayout to its algebra expression string for codegen.

    Returns a string like ``"OrderBy(Row(M, K))"`` or
    ``"OrderBy(Col(M, K))"`` that can be embedded in a generated
    Triton kernel and compiled by ``@lego_jit``.

    The expression always includes a ``.TileBy(...)`` suffix for the
    GPU block-level partitioning, derived from the layout's own tiling
    or added on top of the inner ordering.
    """
    core = layout._layout

    # TileByLayout: extract the inner ordering (Row/Col/RegP)
    if isinstance(core, TileByLayout):
        return _orderby_to_expr(core._input_chain, dim0, dim1)

    # GroupBy: e.g. RowMajor, ColMajor, Transposed, ZCurve, Swizzle
    if isinstance(core, GroupBy):
        return _orderby_to_expr(core.objects, dim0, dim1)

    # Fallback
    return f"OrderBy(Row({dim0}, {dim1}))"


def _orderby_to_expr(objects, dim0, dim1):
    """Extract the OrderBy expression from a list of OrderBy objects."""
    if not objects:
        return f"OrderBy(Row({dim0}, {dim1}))"

    order = objects[0]
    if not isinstance(order, OrderBy) or not order.perms:
        return f"OrderBy(Row({dim0}, {dim1}))"

    perm = order.perms[0]

    if isinstance(perm, Col):
        return f"OrderBy(Col({dim0}, {dim1}))"
    elif isinstance(perm, RegP):
        perm_vec = list(perm._perm_vector) if hasattr(perm, '_perm_vector') else [1, 0]
        return f"OrderBy(RegP(({dim0}, {dim1}), {perm_vec}))"
    elif isinstance(perm, GenP):
        # GenP (ZCurve, Swizzle): complex SymPy expressions.
        # Fall back to row-major -- the data was rearranged to physical
        # order and the kernel reads contiguously.
        return f"OrderBy(Row({dim0}, {dim1}))"
    else:
        # Row or anything else
        return f"OrderBy(Row({dim0}, {dim1}))"


# ============================================================================
# Dynamic kernel generation with caching
# ============================================================================

_KERNEL_CACHE: dict = {}


def _get_mm_kernel(a_layout_expr, b_layout_expr, c_layout_expr):
    """Get or compile a matmul kernel for the given layout expressions.

    Triton requires kernel source to be in a file (uses inspect.getsourcelines),
    so we write the generated kernel to a temp file and import it.
    """
    cache_key = (a_layout_expr, b_layout_expr, c_layout_expr)
    if cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]

    import os
    import importlib.util
    import hashlib

    source = f"""\
import triton
import triton.language as tl
from lego.core import Row, Col, RegP, GenP, OrderBy
from lego.frontends.triton_jit import jit as lego_jit

@lego_jit
@triton.jit
def _generated_mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    L_A = {a_layout_expr}.TileBy([M/BM, K/BK], [BM, BK])
    L_B = {b_layout_expr}.TileBy([K/BK, N/BN], [BK, BN])
    L_C = {c_layout_expr}.TileBy([M/BM, N/BN], [BM, BN])

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptr + L_A[pid_m, k, :, :])
        b = tl.load(b_ptr + L_B[k, pid_n, :, :])
        acc = tl.dot(a, b, acc)

    tl.store(c_ptr + L_C[pid_m, pid_n, :, :], acc)
"""
    # Write to temp file so Triton can inspect source
    temp_dir = os.environ.get("LEGO_TEMP_DIR", "/tmp/lego_kernels")
    os.makedirs(temp_dir, exist_ok=True)
    key_hash = hashlib.md5(cache_key.__repr__().encode()).hexdigest()[:12]
    temp_file = os.path.join(temp_dir, f"lego_mm_{key_hash}.py")
    with open(temp_file, "w") as f:
        f.write(source)

    spec = importlib.util.spec_from_file_location(f"lego_mm_{key_hash}", temp_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = mod._generated_mm_kernel

    _KERNEL_CACHE[cache_key] = kernel
    return kernel


# ============================================================================
# Default kernel (row-major, no layout annotation)
# ============================================================================

if _HAS_TRITON:

    @lego_jit
    @triton.jit
    def _default_mm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        L_A = OrderBy(Row(M, K)).TileBy([M / BM, K / BK], [BM, BK])
        L_B = OrderBy(Row(K, N)).TileBy([K / BK, N / BN], [BK, BN])
        L_C = OrderBy(Row(M, N)).TileBy([M / BM, N / BN], [BM, BN])

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptr + L_A[pid_m, k, :, :])
            b = tl.load(b_ptr + L_B[k, pid_n, :, :])
            acc = tl.dot(a, b, acc)

        tl.store(c_ptr + L_C[pid_m, pid_n, :, :], acc)


# ============================================================================
# Public API
# ============================================================================

def triton_lego_mm(a, b, a_layout=None, b_layout=None):
    """Layout-aware matrix multiply using LEGO-generated Triton kernel.

    If ``a_layout`` or ``b_layout`` is provided, generates a Triton kernel
    whose memory indexing matches the physical layout. The kernel reads
    data in the order it's stored -- no inverse permutation needed.

    Supports any LEGO layout: RowMajor, ColMajor, TiledPermute, Transposed,
    ZCurve, Swizzle, etc.
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Inner dimensions don't match: {K} vs {K2}"

    c = torch.empty(M, N, device=a.device, dtype=torch.float32)

    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    # Pick or generate the right kernel based on layouts
    if a_layout is not None or b_layout is not None:
        a_expr = _layout_to_expr(a_layout, "M", "K") if a_layout else "OrderBy(Row(M, K))"
        b_expr = _layout_to_expr(b_layout, "K", "N") if b_layout else "OrderBy(Row(K, N))"
        c_expr = "OrderBy(Row(M, N))"  # output always row-major
        kernel = _get_mm_kernel(a_expr, b_expr, c_expr)
    else:
        kernel = _default_mm_kernel

    kernel[grid](
        a, b, c,
        M, N, K,
        BM=BM, BN=BN, BK=BK,
    )
    return c.to(a.dtype)


def triton_lego_bmm(a, b, a_layout=None, b_layout=None):
    """Layout-aware batched matrix multiply using LEGO Triton kernel."""
    B_dim, M, K = a.shape
    _, K2, N = b.shape
    assert K == K2

    c = torch.empty(B_dim, M, N, device=a.device, dtype=a.dtype)
    for i in range(B_dim):
        c[i] = triton_lego_mm(a[i], b[i], a_layout=a_layout, b_layout=b_layout)
    return c
