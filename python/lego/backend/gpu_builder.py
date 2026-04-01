"""
Shared GPU kernel builder for all GPU backends (SPIR-V, CUDA, ROCm, etc.).

This is the single source of truth for generating GPU MLIR from LEGO layouts.
All backends use KernelBuilder to produce target-agnostic GPU MLIR, then
apply target-specific lowering via registered GPUTarget descriptors.

Supports:
  - Multiple buffers with different LEGO layouts
  - Shared memory (workgroup memory) with dynamic layout reassignment
  - Barriers (gpu.barrier)
  - Inner tile loops (scf.for with layout-derived indices)
  - Symbolic dimensions
  - User-defined kernel bodies via KernelContext

No GPU hardware required for IR generation.

Adding a new GPU backend
========================
1. Add a C++ pipeline (see LegoNVVMPipeline.cpp / LegoROCDLPipeline.cpp).
2. Register it here::

       GPUTarget(
           name="xevm",
           pipeline="lego-to-xevm",
           default_chip="pvc",
           default_format="assembly",
       ).register()

   That's it — ``builder.compile(target="xevm")`` and
   ``lego.compile(..., target="xevm")`` will work automatically.
"""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from lego.mlir.ir import (
    Context, Location, Module, InsertionPoint,
    IndexType, IntegerType, MemRefType, FunctionType, StringAttr, IntegerAttr,
    F32Type,
)
from lego.mlir.dialects import func as func_dialect
from lego.mlir.dialects import scf as scf_dialect
from lego.mlir.dialects import memref as memref_dialect
from lego.mlir.dialects import arith as arith_dialect
from lego.mlir.dialects import gpu as gpu_dialect
from lego.mlir.passmanager import PassManager

from lego.backend.dialects.lego_dialect import register as register_lego
from lego.backend._ops import (
    _index_const,
    _emit_apply, _emit_apply_inverse,
    _emit_row, _emit_col, _emit_order_by, _emit_group_by,
)
from lego.backend.compiler import (
    DType, _dtype_to_mlir, _get_mlir_element_type, _get_layout_dims,
)
from lego.backend.symbolic import emit_layout_from_python, _resolve_dim


def _ensure_stack_size():
    """Increase stack size for deeply nested LEGO IR (TileBy on large dims)."""
    import sys, resource
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if soft != resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, hard))
    except (ValueError, resource.error):
        pass
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))


# ============================================================================
# Compile result
# ============================================================================

@dataclass
class CompileResult:
    """Result of a multi-target compilation."""
    target: str
    kernel_path: str = ""
    kernel_source: str = ""
    spv_path: str = ""
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# GPU target registry — add new backends here
# ============================================================================

# Global registry: name → GPUTarget
_GPU_TARGETS: Dict[str, "GPUTarget"] = {}


@dataclass
class GPUTarget:
    """Describes a GPU compilation backend.

    Each target maps to a named MLIR pass pipeline (e.g. ``lego-to-nvvm``)
    that is registered on the C++ side.  The Python side only needs this
    descriptor to drive compilation.

    Attributes:
        name:           User-facing target name (e.g. "cuda", "rocm").
        pipeline:       MLIR pipeline name (e.g. "lego-to-nvvm").
        default_chip:   Default chip/arch when the user doesn't specify one.
        default_format: Default binary format ("fatbin", "assembly", …).
        tmp_prefix:     Prefix for temp directories.
    """
    name: str
    pipeline: str
    default_chip: str
    default_format: str
    default_features: Optional[str] = None
    tmp_prefix: str = "lego_gpu_"

    def pipeline_string(self, chip: Optional[str] = None,
                        format: Optional[str] = None,
                        features: Optional[str] = None,
                        opt_level: Optional[int] = None) -> str:
        """Build the ``builtin.module(...)`` pipeline string."""
        chip = chip or self.default_chip
        fmt = format or self.default_format
        features = features or self.default_features
        opts = f"chip={chip} format={fmt}"
        if features is not None:
            opts += f" features={features}"
        if opt_level is not None:
            opts += f" opt-level={opt_level}"
        return f"builtin.module({self.pipeline}{{{opts}}})"

    def register(self):
        """Add this target to the global registry."""
        _GPU_TARGETS[self.name] = self
        return self


# --- Built-in targets ---------------------------------------------------

def _detect_cuda_target():
    """Auto-detect GPU compute capability → (chip, features).

    Returns (sm_XX, +ptxYY) matching the host GPU, or safe defaults.
    """
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            cap = r.stdout.strip().split("\n")[0].strip()
            chip = "sm_" + cap.replace(".", "")
            major = int(cap.split(".")[0])
            ptx = {7: "+ptx70", 8: "+ptx78", 9: "+ptx80",
                   10: "+ptx85", 11: "+ptx86", 12: "+ptx87"}.get(major, "+ptx78")
            return chip, ptx
    except Exception:
        pass
    return "sm_80", "+ptx78"


_cuda_chip, _cuda_features = _detect_cuda_target()

GPUTarget(
    name="cuda",
    pipeline="lego-to-nvvm",
    default_chip=_cuda_chip,
    default_format="fatbin",
    default_features=_cuda_features,
    tmp_prefix="lego_cuda_",
).register()

GPUTarget(
    name="rocm",
    pipeline="lego-to-rocdl",
    default_chip="gfx900",
    default_format="assembly",
    tmp_prefix="lego_rocm_",
).register()

GPUTarget(
    name="llvmspirv",
    pipeline="lego-to-llvmspirv",
    default_chip="generic",
    default_format="assembly",
    tmp_prefix="lego_llvmspirv_",
).register()


# ============================================================================
# Layout-aware buffer descriptor
# ============================================================================

@dataclass
class LayoutBuffer:
    """A buffer with an associated LEGO layout.

    Args:
        layout: LEGO layout object (Row, Col, TileBy, GroupBy, etc.)
        shape:  Tensor shape tuple
        dtype:  Element type (DType.f32, DType.i32, or string "f32", "i32", etc.)
        shared: If True, this buffer is in workgroup (shared) memory.
    """
    layout: object
    shape: Tuple[int, ...]
    dtype: Union[DType, str] = DType.f32
    shared: bool = False

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n


# ============================================================================
# Kernel context — passed to user kernel body functions
# ============================================================================

class _DimAccessor:
    """Attribute-style access to GPU dimension ops: accessor.x, .y, .z."""

    def __init__(self, op):
        self._op = op

    @property
    def x(self):
        return self._op(gpu_dialect.Dimension.x)

    @property
    def y(self):
        return self._op(gpu_dialect.Dimension.y)

    @property
    def z(self):
        return self._op(gpu_dialect.Dimension.z)


class KernelContext:
    """Context object passed to user kernel body functions.

    Provides:
      - Layout-aware load/store via LEGO ops
      - GPU thread/block ID access
      - Layout apply/apply_inverse for index computation
      - Inner loops with layout-derived indices
      - Barriers for shared memory synchronization
      - Arithmetic helpers
    """

    def __init__(self, buf_vals, buf_descs, shared_vals=None):
        self._buf_vals = buf_vals
        self._buf_descs = buf_descs
        self._shared_vals = shared_vals or {}
        # Cached layout IR values per buffer
        self._layout_cache = {}

    # --- GPU thread/block IDs (use as ctx.block_id.x, ctx.thread_id.y, etc.) ---

    @property
    def block_id(self):
        return _DimAccessor(gpu_dialect.block_id)

    @property
    def thread_id(self):
        return _DimAccessor(gpu_dialect.thread_id)

    @property
    def block_dim(self):
        return _DimAccessor(gpu_dialect.block_dim)

    # --- Layout index computation ---

    def apply_inverse(self, layout, flat_idx):
        """Apply layout inverse: flat_idx → multi-dim indices."""
        layout_val = emit_layout_from_python(layout, {})
        dims = _get_layout_dims(layout)
        return _emit_apply_inverse(layout_val, flat_idx, len(dims))

    def apply(self, layout, *indices):
        """Apply layout forward: multi-dim indices → flat index."""
        layout_val = emit_layout_from_python(layout, {})
        return _emit_apply(layout_val, list(indices))

    # --- Layout-aware load/store ---

    def _get_buf_layout_ir(self, buf_index):
        """Get or create MLIR layout IR for a buffer."""
        desc = self._buf_descs[buf_index]
        return emit_layout_from_python(desc.layout, {})

    def load(self, buf_index: int, indices: list):
        """Load from buffer using LEGO layout with explicit multi-dim indices."""
        layout_val = self._get_buf_layout_ir(buf_index)
        memref = self._buf_vals[buf_index]
        flat_idx = _emit_apply(layout_val, indices)
        return memref_dialect.LoadOp(memref, [flat_idx]).result

    def store(self, value, buf_index: int, indices: list):
        """Store to buffer using LEGO layout with explicit multi-dim indices."""
        layout_val = self._get_buf_layout_ir(buf_index)
        memref = self._buf_vals[buf_index]
        flat_idx = _emit_apply(layout_val, indices)
        memref_dialect.StoreOp(value, memref, [flat_idx])

    def load_flat(self, buf_index: int, flat_idx):
        """Load from buffer at a flat index (no layout)."""
        return memref_dialect.LoadOp(self._buf_vals[buf_index], [flat_idx]).result

    def store_flat(self, value, buf_index: int, flat_idx):
        """Store to buffer at a flat index (no layout)."""
        memref_dialect.StoreOp(value, self._buf_vals[buf_index], [flat_idx])

    def set_layout(self, buf_index: int, new_layout):
        """Change the layout of a buffer (for shared memory phase changes)."""
        self._buf_descs[buf_index] = LayoutBuffer(
            layout=new_layout,
            shape=self._buf_descs[buf_index].shape,
            dtype=self._buf_descs[buf_index].dtype,
            shared=self._buf_descs[buf_index].shared,
        )

    # --- Loops ---

    def tile_loop(self, layout, body_fn):
        """Generate an scf.for loop with layout-derived multi-dim indices.

        body_fn receives (indices, flat_idx) where indices is a tuple
        from apply_inverse(layout, flat_idx).
        """
        dims = _get_layout_dims(layout)
        total = 1
        for d in dims:
            total *= int(d)

        loop = scf_dialect.ForOp(
            _index_const(0), _index_const(total), _index_const(1)
        )
        with InsertionPoint(loop.body):
            flat_idx = loop.induction_variable
            indices = self.apply_inverse(layout, flat_idx)
            body_fn(indices, flat_idx)
            scf_dialect.YieldOp([])

    # --- Barriers ---

    def barrier(self):
        """Insert a GPU barrier (thread synchronization)."""
        gpu_dialect.BarrierOp()

    # --- Warp / subgroup operations ---

    def _i32_const(self, value):
        """Create a constant i32 MLIR Value."""
        i32 = IntegerType.get_signless(32)
        return arith_dialect.ConstantOp(i32, IntegerAttr.get(i32, int(value))).result

    def lane_id(self):
        """Return the lane ID within the subgroup (warp)."""
        return gpu_dialect.LaneIdOp().result

    def subgroup_size(self):
        """Return the number of lanes in a subgroup."""
        return gpu_dialect.SubgroupSizeOp().result

    def shuffle_down(self, val, offset, width=32):
        """Shuffle value down by offset within subgroup."""
        from lego.mlir.dialects._gpu_enum_gen import ShuffleMode
        w = self._i32_const(width)
        off = self._i32_const(offset) if isinstance(offset, int) else \
            arith_dialect.IndexCastOp(IntegerType.get_signless(32), offset).result
        return gpu_dialect.ShuffleOp(val, off, w, ShuffleMode.DOWN).shuffleResult

    def shuffle_up(self, val, offset, width=32):
        """Shuffle value up by offset within subgroup."""
        from lego.mlir.dialects._gpu_enum_gen import ShuffleMode
        w = self._i32_const(width)
        off = self._i32_const(offset) if isinstance(offset, int) else \
            arith_dialect.IndexCastOp(IntegerType.get_signless(32), offset).result
        return gpu_dialect.ShuffleOp(val, off, w, ShuffleMode.UP).shuffleResult

    def shuffle_xor(self, val, mask, width=32):
        """Shuffle value with XOR mask within subgroup (butterfly pattern)."""
        from lego.mlir.dialects._gpu_enum_gen import ShuffleMode
        w = self._i32_const(width)
        m = self._i32_const(mask) if isinstance(mask, int) else \
            arith_dialect.IndexCastOp(IntegerType.get_signless(32), mask).result
        return gpu_dialect.ShuffleOp(val, m, w, ShuffleMode.XOR).shuffleResult

    def shuffle_idx(self, val, lane, width=32):
        """Shuffle: get value from specific lane (broadcast pattern)."""
        from lego.mlir.dialects._gpu_enum_gen import ShuffleMode
        w = self._i32_const(width)
        ln = self._i32_const(lane) if isinstance(lane, int) else \
            arith_dialect.IndexCastOp(IntegerType.get_signless(32), lane).result
        return gpu_dialect.ShuffleOp(val, ln, w, ShuffleMode.IDX).shuffleResult

    def subgroup_reduce_add(self, val):
        """Reduce value across all lanes in subgroup with addition."""
        from lego.mlir.dialects._gpu_enum_gen import AllReduceOperation
        return gpu_dialect.SubgroupReduceOp(val, AllReduceOperation.ADD).result

    def subgroup_reduce_mul(self, val):
        """Reduce value across all lanes in subgroup with multiplication."""
        from lego.mlir.dialects._gpu_enum_gen import AllReduceOperation
        return gpu_dialect.SubgroupReduceOp(val, AllReduceOperation.MUL).result

    def subgroup_reduce_max(self, val):
        """Reduce value across all lanes in subgroup with max."""
        from lego.mlir.dialects._gpu_enum_gen import AllReduceOperation
        return gpu_dialect.SubgroupReduceOp(val, AllReduceOperation.MAXNUMF).result

    def subgroup_reduce_min(self, val):
        """Reduce value across all lanes in subgroup with min."""
        from lego.mlir.dialects._gpu_enum_gen import AllReduceOperation
        return gpu_dialect.SubgroupReduceOp(val, AllReduceOperation.MINNUMF).result

    # --- Block-wide operations ---

    def all_reduce_add(self, val):
        """Reduce value across all threads in the workgroup with addition."""
        from lego.mlir.dialects._gpu_enum_gen import AllReduceOperation
        return gpu_dialect.AllReduceOp(val, op=AllReduceOperation.ADD).result

    def all_reduce_mul(self, val):
        """Reduce value across all threads in the workgroup with multiplication."""
        from lego.mlir.dialects._gpu_enum_gen import AllReduceOperation
        return gpu_dialect.AllReduceOp(val, op=AllReduceOperation.MUL).result

    def all_reduce_max(self, val):
        """Reduce value across all threads in the workgroup with max."""
        from lego.mlir.dialects._gpu_enum_gen import AllReduceOperation
        return gpu_dialect.AllReduceOp(val, op=AllReduceOperation.MAXNUMF).result

    def all_reduce_min(self, val):
        """Reduce value across all threads in the workgroup with min."""
        from lego.mlir.dialects._gpu_enum_gen import AllReduceOperation
        return gpu_dialect.AllReduceOp(val, op=AllReduceOperation.MINNUMF).result

    # --- Subgroup broadcast ---

    def subgroup_broadcast(self, val, lane=0):
        """Broadcast value from a specific lane to all lanes in subgroup.

        Implemented via shuffle_idx(val, lane) which is universally supported
        (gpu.subgroup_broadcast lacks NVVM lowering for specific_lane mode).
        """
        return self.shuffle_idx(val, lane)

    # --- Warp prefix sum (built from shuffle_up + add) ---

    def warp_prefix_sum_inclusive(self, val, warp_size=32):
        """Inclusive prefix sum within subgroup via Hillis-Steele shuffle_up.

        Each lane k gets sum(val[0..k]). Built from log2(warp_size)
        shuffle_up + add steps.
        """
        from lego.mlir.dialects._gpu_enum_gen import ShuffleMode
        i32 = IntegerType.get_signless(32)
        result = val
        offset = 1
        while offset < warp_size:
            w = self._i32_const(warp_size)
            off = self._i32_const(offset)
            shuffled = gpu_dialect.ShuffleOp(result, off, w, ShuffleMode.UP)
            neighbor = shuffled.shuffleResult
            valid = shuffled.valid
            # Add neighbor only if valid (lane >= offset)
            added = arith_dialect.AddFOp(result, neighbor).result
            result = arith_dialect.SelectOp(valid, added, result).result
            offset *= 2
        return result

    def warp_prefix_sum_exclusive(self, val, warp_size=32):
        """Exclusive prefix sum: each lane k gets sum(val[0..k-1]), lane 0 gets 0."""
        from lego.mlir.dialects._gpu_enum_gen import ShuffleMode
        # Shift values up by 1 lane, lane 0 gets 0
        i32 = IntegerType.get_signless(32)
        w = self._i32_const(warp_size)
        off = self._i32_const(1)
        shuffled = gpu_dialect.ShuffleOp(val, off, w, ShuffleMode.UP)
        shifted = shuffled.shuffleResult
        valid = shuffled.valid
        zero = arith_dialect.ConstantOp(F32Type.get(), 0.0).result
        result = arith_dialect.SelectOp(valid, shifted, zero).result
        # Now do inclusive prefix sum on the shifted values
        return self.warp_prefix_sum_inclusive(result, warp_size)

    # --- Subgroup MMA operations ---

    def _reshape_buf_2d(self, buf_index, rows, cols):
        """Reinterpret a 1D buffer as a 2D memref for MMA ops."""
        from lego.mlir.ir import MemRefType, DenseI64ArrayAttr
        flat_memref = self._buf_vals[buf_index]
        elem_ty = F32Type.get()
        target_ty = MemRefType.get([rows, cols], elem_ty)
        return memref_dialect.ReinterpretCastOp(
            target_ty, flat_memref,
            offsets=[], sizes=[], strides=[],
            static_offsets=DenseI64ArrayAttr.get([0]),
            static_sizes=DenseI64ArrayAttr.get([rows, cols]),
            static_strides=DenseI64ArrayAttr.get([cols, 1])).result

    def tensor_core(self, tile_m, tile_n, tile_k, a_dtype="f16", c_dtype="f32"):
        """Create a TensorCore handle for MMA operations.

        Usage::

            tc = tensor_core(16, 16, 16)
            acc = tc.zero()
            a = tc.load_a(A2d, row, col, lead_dim)
            b = tc.load_b(B2d, row, col, lead_dim)
            acc = tc.mma(a, b, acc)
            tc.store(acc, C2d, row, col, lead_dim)
        """
        return _TensorCoreHandle(tile_m, tile_n, tile_k, a_dtype, c_dtype)

    # --- Math operations ---

    def exp(self, val):
        """Compute e^val (math.exp)."""
        from lego.mlir.ir import Operation
        return Operation.create("math.exp", results=[val.type], operands=[val]).result

    # --- Arithmetic helpers ---

    def addf(self, a, b):
        return arith_dialect.AddFOp(a, b).result

    def mulf(self, a, b):
        return arith_dialect.MulFOp(a, b).result

    def subf(self, a, b):
        return arith_dialect.SubFOp(a, b).result

    def addi(self, a, b):
        return arith_dialect.AddIOp(a, b).result

    def muli(self, a, b):
        return arith_dialect.MulIOp(a, b).result

    def add(self, a, b):
        return self.addf(a, b)

    def mul(self, a, b):
        return self.mulf(a, b)

    # --- Constants ---

    def const_f32(self, value):
        return arith_dialect.ConstantOp(F32Type.get(), float(value)).result

    def const_index(self, value):
        return _index_const(int(value))

    # --- For-range loop with accumulator ---

    def for_range(self, n, body_fn, init_vals=None):
        """SCF for loop 0..n with optional carried accumulator values."""
        if isinstance(n, int):
            n = _index_const(n)
        if init_vals is None:
            loop = scf_dialect.ForOp(_index_const(0), n, _index_const(1))
            with InsertionPoint(loop.body):
                body_fn(loop.induction_variable)
                scf_dialect.YieldOp([])
            return None
        loop = scf_dialect.ForOp(_index_const(0), n, _index_const(1), init_vals)
        with InsertionPoint(loop.body):
            iv = loop.induction_variable
            carry_args = list(loop.inner_iter_args)
            updated = body_fn(iv, *carry_args)
            scf_dialect.YieldOp(updated)
        return list(loop.results)

    # --- Conditionals ---

    def if_(self, cond, then_fn):
        if_op = scf_dialect.IfOp(cond, has_else=False)
        with InsertionPoint(if_op.then_block):
            then_fn()
            scf_dialect.YieldOp([])

    # --- Comparisons ---

    def lt(self, a, b):
        return arith_dialect.CmpIOp(arith_dialect.CmpIPredicate.ult, a, b).result

    def eq(self, a, b):
        return arith_dialect.CmpIOp(arith_dialect.CmpIPredicate.eq, a, b).result

    def load_raw(self, buf_index, gid=None):
        if gid is None:
            raise ValueError("load_raw requires a gid argument")
        return self.load_flat(buf_index, gid)


class _TensorCoreHandle:
    """Handle for warp-collective MMA operations.

    Wraps gpu.subgroup_mma_* ops with typed fragment management.
    Created via ``ctx.tensor_core(M, N, K, a_dtype, c_dtype)``.
    """

    def __init__(self, tile_m, tile_n, tile_k, a_dtype="f16", c_dtype="f32"):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.a_dtype = a_dtype
        self.c_dtype = c_dtype

    def _mma_type(self, rows, cols, dtype_str, operand):
        from lego.mlir.ir import Type
        return Type.parse(f'!gpu.mma_matrix<{rows}x{cols}x{dtype_str}, "{operand}">')

    def load_a(self, buf_2d, row, col, lead_dim):
        """Load A fragment from 2D memref."""
        mma_type = self._mma_type(self.tile_m, self.tile_k, self.a_dtype, "AOp")
        return gpu_dialect.SubgroupMmaLoadMatrixOp(
            mma_type, buf_2d, [row, col], lead_dim).result

    def load_b(self, buf_2d, row, col, lead_dim):
        """Load B fragment from 2D memref."""
        mma_type = self._mma_type(self.tile_k, self.tile_n, self.a_dtype, "BOp")
        return gpu_dialect.SubgroupMmaLoadMatrixOp(
            mma_type, buf_2d, [row, col], lead_dim).result

    def zero(self):
        """Create zero-initialized accumulator fragment."""
        mma_type = self._mma_type(self.tile_m, self.tile_n, self.c_dtype, "COp")
        zero = arith_dialect.ConstantOp(F32Type.get(), 0.0).result
        return gpu_dialect.SubgroupMmaConstantMatrixOp(mma_type, zero).result

    def mma(self, a_frag, b_frag, c_frag):
        """Compute D = A * B + C (warp-collective MMA)."""
        return gpu_dialect.SubgroupMmaComputeOp(a_frag, b_frag, c_frag).result

    def store(self, frag, buf_2d, row, col, lead_dim):
        """Store result fragment to 2D memref."""
        gpu_dialect.SubgroupMmaStoreMatrixOp(frag, buf_2d, [row, col], lead_dim)



# ============================================================================
# KernelBuilder — shared GPU kernel builder for all backends
# ============================================================================

class KernelBuilder:
    """Build a GPU kernel with multiple layout-aware buffers.

    Supports both simple element-wise kernels and complex tiled kernels
    with shared memory, barriers, and inner loops.

    Example — simple vecadd::

        a = LayoutBuffer(Row(N), shape=(N,))
        b = LayoutBuffer(Row(N), shape=(N,))
        c = LayoutBuffer(Row(N), shape=(N,))

        def vecadd(ctx):
            gid = ctx.global_id()
            va = ctx.load_flat(0, gid)
            vb = ctx.load_flat(1, gid)
            ctx.store_flat(ctx.addf(va, vb), 2, gid)

        builder = KernelBuilder(buffers=[a, b, c], kernel_body=vecadd)

    Example — tiled transpose with shared memory::

        smem = LayoutBuffer(smem_layout, shape=(TD, TD), shared=True)
        builder = KernelBuilder(
            buffers=[src_buf, dst_buf, smem],
            kernel_body=transpose_body,
            grid=(NX//TD * NY//TD, 1, 1),
            block=(TD*TD, 1, 1),
        )
    """

    def __init__(
        self,
        buffers: Sequence[LayoutBuffer],
        kernel_body: Callable[[KernelContext], None],
        name: str = "kernel",
        workgroup_size: int = 64,
        grid: Optional[Tuple[int, ...]] = None,
        block: Optional[Tuple[int, ...]] = None,
    ):
        self._buffers = list(buffers)
        self._kernel_body = kernel_body
        self._name = name

        # Separate global vs shared buffers
        self._global_bufs = [b for b in self._buffers if not b.shared]
        self._shared_bufs = [b for b in self._buffers if b.shared]

        # Grid/block can be explicit or derived from workgroup_size
        if grid is not None and block is not None:
            self._grid = tuple(grid) + (1,) * (3 - len(grid))
            self._block = tuple(block) + (1,) * (3 - len(block))
        else:
            # Derive from first global buffer size
            if self._global_bufs:
                total = self._global_bufs[0].numel
            else:
                total = self._shared_bufs[0].numel
            self._grid = ((total + workgroup_size - 1) // workgroup_size, 1, 1)
            self._block = (workgroup_size, 1, 1)

    def build_module(self):
        """Build target-agnostic MLIR module with gpu.launch + LEGO ops."""
        _ensure_stack_size()
        ctx = Context()
        register_lego(ctx)
        ctx.load_all_available_dialects()

        with ctx, Location.unknown():
            module = Module.create()

            # Function args = global buffers only (shared mem is allocated in kernel)
            memref_types = []
            for buf in self._global_bufs:
                elem_ty = _get_mlir_element_type(ctx, buf.dtype)
                memref_types.append(MemRefType.get([buf.numel], elem_ty))

            with InsertionPoint(module.body):
                self._build_host_func(ctx, memref_types)

        return ctx, module

    def _build_host_func(self, mlir_ctx, memref_types):
        func_ty = FunctionType.get(memref_types, [])
        f = func_dialect.FuncOp(self._name, func_ty)
        f.sym_visibility = StringAttr.get("public")

        entry = f.add_entry_block()
        global_vals = list(entry.arguments)

        with InsertionPoint(entry):
            grid_vals = [_index_const(g) for g in self._grid]
            block_vals = [_index_const(b) for b in self._block]

            launch = gpu_dialect.LaunchOp(
                grid_size=tuple(grid_vals),
                block_size=tuple(block_vals),
            )

            body = launch.body.blocks[0]

            # Add workgroup memory attributions for shared buffers
            # Must be done BEFORE entering the InsertionPoint for body
            if self._shared_bufs:
                launch.attributes["workgroup_attributions"] = IntegerAttr.get(
                    IndexType.get(), len(self._shared_bufs)
                )
                for sbuf in self._shared_bufs:
                    elem_ty = _get_mlir_element_type(mlir_ctx, sbuf.dtype)
                    smem_ty = MemRefType.get(
                        [sbuf.numel], elem_ty,
                        memory_space=IntegerAttr.get(IndexType.get(), 3),
                    )
                    body.add_argument(smem_ty, Location.unknown())

            with InsertionPoint(body):
                # Map each buffer to its MLIR value
                buf_vals = []
                buf_descs = list(self._buffers)
                global_idx = 0
                shared_arg_idx = 12  # after 12 standard gpu.launch args

                for buf in self._buffers:
                    if buf.shared:
                        buf_vals.append(body.arguments[shared_arg_idx])
                        shared_arg_idx += 1
                    else:
                        buf_vals.append(global_vals[global_idx])
                        global_idx += 1

                kctx = KernelContext(buf_vals, buf_descs)
                self._kernel_body(kctx)

                gpu_dialect.TerminatorOp()

            func_dialect.ReturnOp([])

    # --- Convenience: global thread ID for element-wise kernels ---

    @staticmethod
    def global_id_1d(ctx):
        """Compute 1D global thread ID = block_id_x * block_dim_x + thread_id_x."""
        bid = ctx.block_id.x
        bdim = ctx.block_dim.x
        tid = ctx.thread_id.x
        return arith_dialect.AddIOp(
            arith_dialect.MulIOp(bid, bdim).result, tid
        ).result

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self, target: str = "webgpu", **kwargs) -> CompileResult:
        """Compile the kernel to the given target.

        Args:
            target: Any registered GPU target ("cuda", "rocm", …) or
                    SPIR-V target ("vulkan", "webgpu", "metal", "webgl").
            **kwargs: Forwarded to the backend (chip, format, output_dir, …).
        """
        if target in _GPU_TARGETS:
            return self._compile_gpu_target(_GPU_TARGETS[target], **kwargs)
        from lego.backend.spirv import compile_to_target
        return compile_to_target(self, target=target, **kwargs)

    def _compile_gpu_target(self, gpu_target: GPUTarget, output_dir=None,
                            name=None, chip=None, format=None,
                            **kwargs) -> CompileResult:
        """Generic GPU compilation via a registered MLIR pipeline."""
        ctx, module = self.build_module()
        if name is None:
            name = self._name

        pipeline_str = gpu_target.pipeline_string(
            chip=chip, format=format, **kwargs)
        with ctx:
            try:
                pm = PassManager.parse(pipeline_str)
                pm.run(module.operation)
            except Exception as e:
                raise RuntimeError(
                    f"{gpu_target.name} compilation failed "
                    f"({gpu_target.pipeline}):\n{e}"
                ) from e

        lowered_ir = str(module)
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix=gpu_target.tmp_prefix)
        os.makedirs(output_dir, exist_ok=True)

        out_path = os.path.join(output_dir, f"{name}.mlir")
        Path(out_path).write_text(lowered_ir)

        return CompileResult(
            target=gpu_target.name,
            kernel_path=out_path,
            kernel_source=lowered_ir,
            metadata={
                "pipeline": gpu_target.pipeline,
                "chip": chip or gpu_target.default_chip,
                "format": format or gpu_target.default_format,
            },
        )


# ============================================================================
# Convenience: simple element-wise kernels
# ============================================================================

def make_elementwise_kernel(
    buffers: Sequence[LayoutBuffer],
    body_fn: Callable,
    name: str = "kernel",
    workgroup_size: int = 64,
) -> KernelBuilder:
    """Create a KernelBuilder where each thread handles one element.

    body_fn receives (ctx, gid) where gid is the global flat index.
    Includes automatic bounds checking.
    """
    total = buffers[0].numel

    def wrapped_body(ctx):
        gid = KernelBuilder.global_id_1d(ctx)
        in_bounds = arith_dialect.CmpIOp(
            arith_dialect.CmpIPredicate.ult, gid, _index_const(total)
        )
        if_op = scf_dialect.IfOp(in_bounds, has_else=False)
        with InsertionPoint(if_op.then_block):
            body_fn(ctx, gid)
            scf_dialect.YieldOp([])

    return KernelBuilder(
        buffers=buffers, kernel_body=wrapped_body,
        name=name, workgroup_size=workgroup_size,
    )


def make_permutation_kernel(
    layout, shape, dtype="f32", workgroup_size=64
) -> KernelBuilder:
    """Create a kernel that permutes data according to a LEGO layout.

    Equivalent to: dst[identity(inv(layout, gid))] = src[gid]
    """
    src = LayoutBuffer(layout, shape, dtype)
    dst = LayoutBuffer(layout, shape, dtype)

    def permute_body(ctx, gid):
        val = memref_dialect.LoadOp(ctx._buf_vals[0], [gid]).result
        layout_dims = _get_layout_dims(layout)
        dim_vals = [_resolve_dim(d, {}) for d in layout_dims]
        rank = len(layout_dims)
        identity = _emit_group_by(
            dim_vals, [_emit_order_by([_emit_row(dim_vals)])]
        )
        layout_val = emit_layout_from_python(layout, {})
        indices = _emit_apply_inverse(layout_val, gid, rank)
        out_idx = _emit_apply(identity, indices)
        memref_dialect.StoreOp(val, ctx._buf_vals[1], [out_idx])

    return make_elementwise_kernel(
        buffers=[src, dst], body_fn=permute_body,
        name="transform", workgroup_size=workgroup_size,
    )
