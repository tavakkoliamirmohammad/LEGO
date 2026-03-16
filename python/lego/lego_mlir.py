from .lego import *
import functools
import os
import sys

from mlir.dialects import arith, scf, func, memref
try:
    from mlir.dialects import affine
except ImportError:
    affine = None
try:
    from mlir.dialects import gpu
except ImportError:
    gpu = None
from mlir.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    IntegerType,
    IntegerAttr
)
from enum import Enum
from mlir import ir
from mlir.ir import *
import mlir.extras.types as T


class MemorySpace(Enum):
    HOST_MEMORY = 0
    GLOBAL_MEMORY = 1
    SHARED_MEMORY = 3
    PRIVATE_MEMORY = 5


class MLIRPrinter:
    """MLIR code generation helper for GPU kernels.

    Provides module wrapping, GPU kernel launch boilerplate, and barriers.
    All index computation goes through LEGO dialect ops (see mlir_apply,
    mlir_apply_inverse, mlir_load, mlir_store, mlir_loop below).
    """

    def __init__(self, ctx=Context()):
        self.ctx = ctx
        self.ctx.allow_unregistered_dialects = True
        try:
            from lego.dialects.lego_dialect import register as _reg
            _reg(self.ctx)
        except Exception:
            pass

    def generate_mlir(self, schedule=None):
        def decorator_body(body):
            @functools.wraps(body)
            def wrapper():
                with self.ctx, Location.unknown():
                    module = Module.create()
                    with InsertionPoint(module.body):
                        @func.FuncOp.from_py_func()
                        def main():
                            return body()
                    if schedule:
                        transform_module = Module.parse(schedule)
                        print(transform_module)
                    print(module)
            return wrapper()
        return decorator_body

    @staticmethod
    def insert_barrier():
        gpu.barrier()

    @staticmethod
    def get_token_type():
        return ir.Type.parse("!gpu.async.token")

    @staticmethod
    def generate_gpu_kernel(ins: List['MLIRTensor'], outs: List['MLIRTensor'], gridSize, blockSize, workgroup_memory=[], private_memory=[]):
        def decorator_body(body):
            @functools.wraps(body)
            def wrapper():
                token_ty = MLIRPrinter.get_token_type()
                token = gpu.wait([])
                for i in set(ins + outs):
                    token = i.gpu_allocate(token)
                    i.host_allocate()
                for i in ins:
                    i.fill_host()
                    token = i.copy_to_device(token)
                gpu.wait([token])

                launch_op = gpu.LaunchOp(
                    list(map(arith.ConstantOp.create_index, gridSize)),
                    list(map(arith.ConstantOp.create_index, blockSize)),
                    async_dependencies=[]
                )
                launch_op.attributes["workgroup_attributions"] = IntegerAttr.get(
                    T.i64(), len(workgroup_memory))

                block = launch_op.body.blocks[0]
                for w in workgroup_memory:
                    block.add_argument(w.get_memref_type_address_space(3), Location.unknown())
                for p in private_memory:
                    block.add_argument(p.get_memref_type_address_space(5), Location.unknown())

                with InsertionPoint(block):
                    for i in set(ins + outs):
                        i.set_memory_space(MemorySpace.GLOBAL_MEMORY)
                        memref.assume_alignment(i.gpu_alloc_ref, 128)
                    for i in range(len(workgroup_memory)):
                        workgroup_memory[i].shared_memory_ref = launch_op.body.blocks[0].arguments[12 + i]
                        workgroup_memory[i].set_memory_space(MemorySpace.SHARED_MEMORY)
                    for i in private_memory:
                        i.set_memory_space(MemorySpace.PRIVATE_MEMORY)
                    body(launch_op.body.blocks[0].arguments)
                    gpu.terminator()

                for i in set(ins + outs):
                    token = i.dealloc_gpu(token)
            return wrapper()
        return decorator_body


class MLIRTensor:
    """Tensor backed by a memref with a LEGO layout.

    Use mlir_load / mlir_store for indexed access (no SymPy).
    """

    def __init__(self, layout: 'GroupBy', dtype="", is_dim_shape=False) -> None:
        self.layout = layout
        self.alloc_ref = None
        self.gpu_alloc_ref = None
        self.shared_memory_ref = None
        self.private_memory_ref = None
        self.data_type = None
        self.is_dim_shape = is_dim_shape
        self.dimension = layout.d
        self.memory_space = MemorySpace.HOST_MEMORY
        if dtype == "f32":
            self.data_type = T.f32()
        if dtype == "f16":
            self.data_type = T.f16()

    def get_memref_type(self) -> memref.MemRefType:
        return self.get_memref_type_address_space(self.memory_space)

    def get_flattend_shape(self):
        return product(self.layout.dims())

    def get_memref_type_address_space(self, address_space) -> memref.MemRefType:
        return T.memref(self.get_flattend_shape(), self.data_type, memory_space=address_space)

    def host_allocate(self):
        self.alloc_ref = memref.alloc(
            self.get_memref_type_address_space(0), [], [])
        return self

    def dealloc_gpu(self, *tokens):
        if tokens is None:
            gpu.dealloc(self.gpu_alloc_ref)
            return None
        token_ty = Type.parse("!gpu.async.token")
        return gpu.dealloc(token_ty, list(tokens), self.gpu_alloc_ref)

    def set_memory_space(self, memory_space: MemorySpace):
        self.memory_space = memory_space
        return self

    def gpu_allocate(self, *tokens):
        if tokens is None:
            self.gpu_alloc_ref = gpu.alloc(
                self.get_memref_type(), [], [], [], [])
            return None
        token_ty = Type.parse("!gpu.async.token")
        tmp = gpu.alloc(
            self.get_memref_type_address_space(0), token_ty, list(tokens), [], [])
        self.gpu_alloc_ref = tmp[0]
        return tmp[1]

    def fill_host(self):
        from mlir.dialects import scf as _scf
        for_op = _scf.ForOp(
            _index_const(0),
            _index_const(int(self.get_flattend_shape())),
            _index_const(1))
        with InsertionPoint(for_op.body):
            i = for_op.induction_variable
            f_i = arith.sitofp(self.data_type, arith.index_cast(T.i32(), i))
            self.store_physical_1d([i], f_i)
            _scf.YieldOp([])
        return self

    def store_physical_1d(self, coords, value):
        return memref.store(value, self.get_memory_ref_address_space(), coords)

    def copy_to_device(self, token):
        token_ty = Type.parse("!gpu.async.token")
        if token is None:
            gpu.memcpy(None, [], self.gpu_alloc_ref, self.alloc_ref)
            return None
        return gpu.memcpy(token_ty, [token], self.gpu_alloc_ref, self.alloc_ref)

    def get_memory_ref_address_space(self):
        memory_space = self.memory_space
        if memory_space == MemorySpace.SHARED_MEMORY:
            return self.shared_memory_ref
        if memory_space == MemorySpace.PRIVATE_MEMORY:
            return self.private_memory_ref
        if memory_space == MemorySpace.GLOBAL_MEMORY:
            return self.gpu_alloc_ref
        return self.alloc_ref


printer = MLIRPrinter()


# ============================================================================
# Pure-MLIR layout helpers (no SymPy dependency)
#
# For use in GPU kernels and benchmarks where all dimensions are concrete
# integers and index computation can go directly through LEGO dialect ops.
# ============================================================================

def mlir_layout(layout):
    """Emit a Python layout object as MLIR LEGO dialect ops.

    Only works when all dimensions are concrete (int / sp.Integer).
    Returns an MLIR Value of !lego.layout type.
    """
    return emit_layout_from_python(layout, {})


def mlir_apply(layout, indices):
    """Forward-apply a layout to MLIR index values.

    Args:
        layout: Python LayoutBlock (OrderBy, TileByLayout, GroupBy, ...)
        indices: list of MLIR Values (index type)
    Returns:
        MLIR Value (index type) — the flat index
    """
    from lego.dialects.lego_dialect import ApplyOp as _ApplyOp
    layout_val = mlir_layout(layout)
    return _ApplyOp(
        flat_index=ir.IndexType.get(),
        layout=layout_val,
        indices=indices,
    ).result


def mlir_apply_inverse(layout, flat_idx):
    """Inverse-apply a layout to an MLIR flat index.

    Args:
        layout: Python LayoutBlock
        flat_idx: MLIR Value (index type)
    Returns:
        list of MLIR Values (index type) — the multi-dim indices
    """
    from lego.dialects.lego_dialect import ApplyInverseOp as _ApplyInverseOp
    layout_val = mlir_layout(layout)
    dims = layout._dims if hasattr(layout, '_dims') else layout.dims()
    rank = len(dims)
    inv = _ApplyInverseOp(
        indices=[ir.IndexType.get()] * rank,
        layout=layout_val,
        flat_index=flat_idx,
    )
    return list(inv.results)


def mlir_cast_view(tensor):
    """Create a lego.view from an MLIRTensor's memref and layout.

    Returns an MLIR Value of !lego.view type.
    """
    from lego.dialects.lego_dialect import CastViewOp as _CastViewOp
    layout_val = mlir_layout(tensor.layout)
    view_ty = ir.Type.parse(f"!lego.view<{tensor.data_type}>")
    return _CastViewOp(
        view=view_ty,
        memref=tensor.get_memory_ref_address_space(),
        layout=layout_val,
    ).result


def mlir_load(tensor, indices):
    """Load from an MLIRTensor using lego.cast_view + lego.load.

    Args:
        tensor: MLIRTensor instance
        indices: list of MLIR Values (index type)
    Returns:
        MLIR Value — the loaded element
    """
    from lego.dialects.lego_dialect import LoadOp as _LoadOp
    view = mlir_cast_view(tensor)
    return _LoadOp(result=tensor.data_type, view=view, indices=indices).result


def mlir_store(value, tensor, indices):
    """Store to an MLIRTensor using lego.cast_view + lego.store.

    Args:
        value: MLIR Value to store
        tensor: MLIRTensor instance
        indices: list of MLIR Values (index type)
    """
    from lego.dialects.lego_dialect import StoreOp as _StoreOp
    view = mlir_cast_view(tensor)
    _StoreOp(value=value, view=view, indices=indices)


def mlir_loop(layout, body_fn):
    """Generate an scf.for loop with LEGO apply_inverse (no SymPy).

    The body_fn receives (indices, induction_var) where indices is
    a list of MLIR Values from the layout inverse.
    """
    from mlir.dialects import scf
    total = 1
    dims = layout._dims if hasattr(layout, '_dims') else layout.dims()
    for d in dims:
        total *= int(d)
    for_op = scf.ForOp(_index_const(0), _index_const(total), _index_const(1))
    with InsertionPoint(for_op.body):
        idx = for_op.induction_variable
        indices = mlir_apply_inverse(layout, idx)
        body_fn(indices, idx)
        scf.YieldOp([])


# ============================================================================
# MLIR Roundtrip: replace simplify_ops (SymPy+z3) with MLIR pipeline
# ============================================================================

from mlir.passmanager import PassManager as _PassManager
from mlir.dialects import func as _func_dialect
from lego.dialects.lego_dialect import (
    register as _register_lego,
    RegPOp, RowOp, ColOp, OrderByOp, GroupByOp, ApplyOp,
    ApplyInverseOp, TileByOp, GenPOp, YieldOp,
)
from lego.dialects._lego_ops_gen import assume_bounds as _assume_bounds_fn


def _index_const(val):
    """Emit arith.constant with index type."""
    from mlir.ir import IndexType, IntegerAttr
    idx_ty = IndexType.get()
    return arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, int(val))).result


def _layout_type():
    """Get !lego.layout type from current context."""
    return ir.Type.parse("!lego.layout")


def _resolve_dim(dim, sym_to_val):
    """Resolve a dimension (int or SymPy expr) to an MLIR Value."""
    if isinstance(dim, int):
        return _index_const(dim)
    if isinstance(dim, sp.Integer):
        return _index_const(int(dim))
    if isinstance(dim, sp.Symbol):
        if dim in sym_to_val:
            return sym_to_val[dim]
        raise KeyError(f"Symbol {dim} not in sym_to_val mapping")
    if isinstance(dim, sp.Expr):
        # For compound expressions like M/BM, lower them to arith ops
        return _lower_sympy_to_index(dim, sym_to_val)
    raise TypeError(f"Cannot resolve dim of type {type(dim)}: {dim}")


def _lower_sympy_cond_to_i1(cond, sym_to_val):
    """Lower a SymPy relational to an MLIR i1 value."""
    if cond is sp.true or cond == True:
        return arith.ConstantOp(ir.IntegerType.get_signless(1),
                                ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 1)).result
    if cond is sp.false or cond == False:
        return arith.ConstantOp(ir.IntegerType.get_signless(1),
                                ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 0)).result

    _PRED_MAP = {
        sp.StrictGreaterThan: arith.CmpIPredicate.sgt,
        sp.GreaterThan: arith.CmpIPredicate.sge,
        sp.StrictLessThan: arith.CmpIPredicate.slt,
        sp.LessThan: arith.CmpIPredicate.sle,
        sp.Equality: arith.CmpIPredicate.eq,
        sp.Unequality: arith.CmpIPredicate.ne,
    }
    pred = _PRED_MAP.get(type(cond))
    if pred is None:
        raise NotImplementedError(f"Unsupported condition: {cond}")
    lhs = _lower_sympy_to_index(cond.lhs, sym_to_val)
    rhs = _lower_sympy_to_index(cond.rhs, sym_to_val)
    return arith.cmpi(pred, lhs, rhs)


def _lower_sympy_to_index(expr, sym_to_val):
    """Lower a SymPy expression to MLIR index-typed arith ops."""
    if isinstance(expr, sp.Integer) or isinstance(expr, int):
        return _index_const(int(expr))
    if isinstance(expr, sp.Symbol):
        if expr in sym_to_val:
            return sym_to_val[expr]
        raise KeyError(f"Symbol {expr} not found")

    if isinstance(expr, sp.Add):
        vals = [_lower_sympy_to_index(a, sym_to_val) for a in expr.args]
        acc = vals[0]
        for v in vals[1:]:
            acc = arith.addi(acc, v)
        return acc

    if isinstance(expr, sp.Mul):
        # Check for division: SymPy represents a/b as a * b^(-1)
        num, den = expr.as_numer_denom()
        if den != sp.S.One:
            # This is a division — lower as divui
            a = _lower_sympy_to_index(num, sym_to_val)
            b = _lower_sympy_to_index(den, sym_to_val)
            return arith.divui(a, b)
        # Pure multiplication
        coeff, rest = expr.as_coeff_Mul()
        if rest == sp.S.One:
            return _index_const(int(coeff))
        vals = []
        for a in expr.args:
            vals.append(_lower_sympy_to_index(a, sym_to_val))
        acc = vals[0]
        for v in vals[1:]:
            acc = arith.muli(acc, v)
        return acc

    if isinstance(expr, sp.floor):
        inner = expr.args[0]
        num, den = inner.as_numer_denom()
        a = _lower_sympy_to_index(num, sym_to_val)
        b = _lower_sympy_to_index(den, sym_to_val)
        return arith.divui(a, b)

    if isinstance(expr, sp.Mod):
        a = _lower_sympy_to_index(expr.args[0], sym_to_val)
        b = _lower_sympy_to_index(expr.args[1], sym_to_val)
        return arith.remui(a, b)

    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if exp == sp.S.NegativeOne:
            raise NotImplementedError(f"Negative power in index expr: {expr}")
        if exp.is_Integer and int(exp) >= 0:
            b = _lower_sympy_to_index(base, sym_to_val)
            result = _index_const(1)
            for _ in range(int(exp)):
                result = arith.muli(result, b)
            return result

    # Max(a, b) -> select(a >= b, a, b)
    if expr.func == sp.Max:
        a = _lower_sympy_to_index(expr.args[0], sym_to_val)
        b = _lower_sympy_to_index(expr.args[1], sym_to_val)
        cmp = arith.cmpi(arith.CmpIPredicate.sge, a, b)
        return arith.select(cmp, a, b)

    # Min(a, b) -> select(a <= b, a, b)
    if expr.func == sp.Min:
        a = _lower_sympy_to_index(expr.args[0], sym_to_val)
        b = _lower_sympy_to_index(expr.args[1], sym_to_val)
        cmp = arith.cmpi(arith.CmpIPredicate.sle, a, b)
        return arith.select(cmp, a, b)

    # Relational -> arith.cmpi producing i1
    if isinstance(expr, sp.core.relational.Relational):
        return _lower_sympy_cond_to_i1(expr, sym_to_val)

    # Piecewise((val1, cond1), (val2, cond2), ..., (valN, True))
    # -> nested scf.if chains
    if isinstance(expr, sp.Piecewise):
        from mlir.dialects import scf as _scf
        idx_ty = ir.IndexType.get()

        def _recurse(i):
            val_expr, cond_expr = expr.args[i]
            then_val = _lower_sympy_to_index(val_expr, sym_to_val)
            # Last clause is the "else" (cond is True)
            if i == len(expr.args) - 1:
                return then_val
            cond_val = _lower_sympy_cond_to_i1(cond_expr, sym_to_val)
            ifop = _scf.IfOp(cond_val, [idx_ty], has_else=True)
            with InsertionPoint(ifop.then_block):
                _scf.YieldOp([then_val])
            with InsertionPoint(ifop.else_block):
                _scf.YieldOp([_recurse(i + 1)])
            return ifop.results[0]

        return _recurse(0)

    raise NotImplementedError(f"Cannot lower SymPy expr to index: {expr}")


def emit_layout_from_python(layout, sym_to_val):
    """Convert a Python layout object to MLIR LEGO dialect ops.

    Args:
        layout: Python LayoutBlock (RegP, OrderBy, GroupBy, GenP)
        sym_to_val: dict {sp.Symbol | int -> MLIR Value}
    Returns:
        MLIR Value of !lego.layout type
    """
    from mlir.ir import IndexType
    lt = _layout_type()

    if isinstance(layout, RegP):
        perm_vec = layout._perm_vector
        dim_vals = [_resolve_dim(d, sym_to_val) for d in layout._dims]
        return RegPOp(result=lt, perm=perm_vec, dims=dim_vals).result

    if isinstance(layout, OrderBy):
        perm_vals = []
        for p in layout.perms:
            perm_vals.append(emit_layout_from_python(p, sym_to_val))
        return OrderByOp(result=lt, perms=perm_vals).result

    if isinstance(layout, TileByLayout):
        all_perm_vals = []
        for orderby in layout._input_chain:
            for p in orderby.perms:
                all_perm_vals.append(emit_layout_from_python(p, sym_to_val))
        input_val = OrderByOp(result=lt, perms=all_perm_vals).result
        tile_dim_vals = [_resolve_dim(d, sym_to_val)
                         for g in layout._tile_groups for d in g]
        return TileByOp(result=lt, input=input_val,
                        tile_dims=tile_dim_vals,
                        tile_shape=layout.tile_shape).result

    if isinstance(layout, GroupBy):
        dim_vals = [_resolve_dim(d, sym_to_val) for d in layout._dims]
        obj_vals = []
        for obj in layout.objects:
            obj_vals.append(emit_layout_from_python(obj, sym_to_val))
        return GroupByOp(result=lt, group_dims=dim_vals, objects=obj_vals).result

    if isinstance(layout, GenP):
        dim_vals = [_resolve_dim(d, sym_to_val) for d in layout._dims]
        idx_ty = IndexType.get()
        gen_p_op = GenPOp(result=lt, dims=dim_vals)

        # Apply region: block args = N indices -> yield single flat index
        rank = len(layout._dims)
        apply_block = gen_p_op.body.blocks.append(*([idx_ty] * rank))
        with InsertionPoint(apply_block):
            temp_syms = [sp.Symbol(f"_genp_arg_{k}", integer=True) for k in range(rank)]
            local_map = dict(sym_to_val)
            for s, arg in zip(temp_syms, apply_block.arguments):
                local_map[s] = arg
            sympy_result = layout.f_apply(tuple(temp_syms))
            mlir_result = _lower_sympy_to_index(sympy_result, local_map)
            YieldOp(values=[mlir_result])

        # Inverse region: block arg = single flat index -> yield N indices
        # (skip if f_inv is not provided — forward-only GenP)
        if layout.f_inv is not None:
            inv_block = gen_p_op.inv_body.blocks.append(idx_ty)
            with InsertionPoint(inv_block):
                temp_flat = sp.Symbol("_genp_flat", integer=True)
                local_map = dict(sym_to_val)
                local_map[temp_flat] = inv_block.arguments[0]
                sympy_inv = layout.f_inv(temp_flat)
                inv_results = [_lower_sympy_to_index(r, local_map) for r in sympy_inv]
                YieldOp(values=inv_results)

        return gen_p_op.result

    raise TypeError(f"Unsupported layout type: {type(layout).__name__}")


def arith_to_sympy(value, val_to_sym, memo=None):
    """Convert an MLIR Value (arith ops after lowering) back to SymPy.

    Args:
        value: MLIR Value to convert
        val_to_sym: dict {MLIR block arg -> sp.Symbol}
        memo: memoization dict keyed by MLIR Value (uses Value.__hash__)
    Returns:
        SymPy expression
    """
    if memo is None:
        memo = {}

    if value in memo:
        return memo[value]

    # Block argument -> look up in val_to_sym
    if value in val_to_sym:
        result = val_to_sym[value]
        memo[value] = result
        return result

    # Check if this is a block argument (no defining op)
    if isinstance(value.owner, ir.Block):
        raise KeyError(f"Block argument not found in val_to_sym mapping")

    # Must have a defining op
    op = value.owner
    op_name = op.name

    if op_name == "arith.constant":
        attr = op.attributes["value"]
        result = sp.Integer(int(ir.IntegerAttr(attr).value))
        memo[value] = result
        return result

    # Binary arith ops → SymPy
    _BINARY_OPS = {
        "arith.addi": lambda a, b: a + b,
        "arith.muli": lambda a, b: a * b,
        "arith.subi": lambda a, b: a - b,
        "arith.divui": lambda a, b: sp.floor(a / b),
        "arith.remui": lambda a, b: sp.Mod(a, b),
    }
    if op_name in _BINARY_OPS:
        a = arith_to_sympy(op.operands[0], val_to_sym, memo)
        b = arith_to_sympy(op.operands[1], val_to_sym, memo)
        result = _BINARY_OPS[op_name](a, b)
        memo[value] = result
        return result

    if op_name == "arith.cmpi":
        pred_attr = op.attributes["predicate"]
        pred = int(ir.IntegerAttr(pred_attr).value)
        a = arith_to_sympy(op.operands[0], val_to_sym, memo)
        b = arith_to_sympy(op.operands[1], val_to_sym, memo)
        # MLIR CmpIPredicate: eq=0, ne=1, slt=2, sle=3, sgt=4, sge=5, ult=6, ule=7, ugt=8, uge=9
        pred_map = {
            0: sp.Eq, 1: sp.Ne,
            2: sp.StrictLessThan, 3: sp.LessThan,
            4: sp.StrictGreaterThan, 5: sp.GreaterThan,
            6: sp.StrictLessThan, 7: sp.LessThan,
            8: sp.StrictGreaterThan, 9: sp.GreaterThan,
        }
        rel_cls = pred_map.get(pred)
        if rel_cls:
            result = rel_cls(a, b, evaluate=False)
            memo[value] = result
            return result
        raise NotImplementedError(f"Unknown cmpi predicate: {pred}")

    if op_name == "arith.select":
        # operands: condition, true_val, false_val
        true_val = arith_to_sympy(op.operands[1], val_to_sym, memo)
        false_val = arith_to_sympy(op.operands[2], val_to_sym, memo)
        # Try to recover Max/Min from the cmpi pattern
        cond_op = op.operands[0].owner
        if cond_op.name == "arith.cmpi":
            pred_attr = cond_op.attributes["predicate"]
            pred = int(ir.IntegerAttr(pred_attr).value)
            cmp_lhs = arith_to_sympy(cond_op.operands[0], val_to_sym, memo)
            cmp_rhs = arith_to_sympy(cond_op.operands[1], val_to_sym, memo)
            # (predicate, lhs_is_true_val) -> sympy function
            # select(sge(a,b), a, b) => Max(a,b)
            # select(sge(a,b), b, a) => Min(a,b)
            # select(sle(a,b), a, b) => Min(a,b)
            # select(sle(a,b), b, a) => Max(a,b)
            select_patterns = {
                (5, True): sp.Max,   # sge, true_val==cmp_lhs
                (5, False): sp.Min,  # sge, true_val==cmp_rhs
                (3, True): sp.Min,   # sle, true_val==cmp_lhs
                (3, False): sp.Max,  # sle, true_val==cmp_rhs
            }

            if true_val == cmp_lhs and false_val == cmp_rhs:
                key = (pred, True)
            elif true_val == cmp_rhs and false_val == cmp_lhs:
                key = (pred, False)
            else:
                key = None

            if key and key in select_patterns:
                result = select_patterns[key](cmp_lhs, cmp_rhs)
                memo[value] = result
                return result
        # Fallback: Piecewise
        cond = arith_to_sympy(op.operands[0], val_to_sym, memo)
        result = sp.Piecewise((true_val, cond), (false_val, True))
        memo[value] = result
        return result

    if op_name == "scf.if":
        # scf.if has two regions: then and else.
        # Reconstruct as Piecewise(then_val, cond), (else_val, True)).
        cond = arith_to_sympy(op.operands[0], val_to_sym, memo)
        then_block = op.regions[0].blocks[0]
        else_block = op.regions[1].blocks[0]
        then_yield = list(then_block)[-1]  # scf.yield
        else_yield = list(else_block)[-1]
        then_val = arith_to_sympy(then_yield.operands[0], val_to_sym, memo)
        else_val = arith_to_sympy(else_yield.operands[0], val_to_sym, memo)
        result = sp.Piecewise((then_val, cond), (else_val, True))
        memo[value] = result
        return result

    raise NotImplementedError(f"Cannot convert op '{op_name}' to SymPy")


def _collect_free_symbols(layout):
    """Collect all SymPy symbols used in a layout's dimensions (recursive)."""
    syms = set()
    if hasattr(layout, '_dims'):
        for d in layout._dims:
            if isinstance(d, sp.Expr):
                syms |= d.free_symbols
            elif isinstance(d, sp.Symbol):
                syms.add(d)
    if isinstance(layout, TileByLayout):
        for orderby in layout._input_chain:
            syms |= _collect_free_symbols(orderby)
        for g in layout._tile_groups:
            for d in g:
                if isinstance(d, sp.Expr):
                    syms |= d.free_symbols
                elif isinstance(d, sp.Symbol):
                    syms.add(d)
    elif isinstance(layout, OrderBy):
        for p in layout.perms:
            syms |= _collect_free_symbols(p)
    elif isinstance(layout, GroupBy):
        for obj in layout.objects:
            syms |= _collect_free_symbols(obj)
    return syms


_LEGO_DEBUG = os.environ.get("LEGO_DEBUG", "")


def simplify_via_mlir(layout, mode, args, constraints=None):
    """Compute layout.apply or layout.inv via MLIR roundtrip.

    Set LEGO_DEBUG=1 to print MLIR IR before and after lowering.

    Args:
        layout: GroupBy or other LayoutBlock
        mode: 'apply' | 'inv'
        args: for 'apply': list of SymPy symbols/exprs (N-D indices)
              for 'inv': single SymPy symbol/expr (flat index)
        constraints: dict {sp.Symbol: (lb, ub)} for assume_bounds
                     lb/ub can be int, sp.Symbol, sp.Expr, or None
    Returns:
        'apply': single SymPy expression (flat index)
        'inv': list of SymPy expressions (N-D indices)
    """
    from mlir.ir import IndexType, FunctionType, StringAttr

    if constraints is None:
        constraints = {}

    # Collect all unique SymPy symbols
    all_syms = set()
    all_syms |= _collect_free_symbols(layout)

    if mode == 'apply':
        for a in args:
            if isinstance(a, sp.Expr):
                all_syms |= a.free_symbols
            elif isinstance(a, sp.Symbol):
                all_syms.add(a)
    else:  # inv
        if isinstance(args, sp.Expr):
            all_syms |= args.free_symbols
        elif isinstance(args, sp.Symbol):
            all_syms.add(args)

    for sym in constraints:
        if isinstance(sym, sp.Symbol):
            all_syms.add(sym)
        lb, ub = constraints[sym]
        if isinstance(lb, sp.Expr):
            all_syms |= lb.free_symbols
        if isinstance(ub, sp.Expr):
            all_syms |= ub.free_symbols

    # Order symbols deterministically
    sym_list = sorted(all_syms, key=lambda s: s.name)

    ctx = Context()
    _register_lego(ctx)

    with ctx, Location.unknown():
        module = Module.create()
        idx_ty = IndexType.get()

        # Determine function signature
        n_args = len(sym_list)
        if mode == 'apply':
            rank = len(layout._dims)
            func_ty = FunctionType.get([idx_ty] * n_args, [idx_ty])
        else:
            rank = len(layout._dims)
            func_ty = FunctionType.get([idx_ty] * n_args, [idx_ty] * rank)

        with InsertionPoint(module.body):
            f = _func_dialect.FuncOp("roundtrip", func_ty)
            f.sym_visibility = StringAttr.get("public")

        entry = f.add_entry_block()

        # Build sym_to_val and val_to_sym mappings
        sym_to_val = {}
        val_to_sym = {}
        for i, sym in enumerate(sym_list):
            sym_to_val[sym] = entry.arguments[i]
            val_to_sym[entry.arguments[i]] = sym

        with InsertionPoint(entry):
            # Emit assume_bounds
            for sym, (lb, ub) in constraints.items():
                if not isinstance(sym, sp.Symbol):
                    continue
                if sym not in sym_to_val:
                    continue
                val = sym_to_val[sym]
                lb_val = _resolve_dim(lb, sym_to_val) if lb is not None else None
                ub_val = _resolve_dim(ub, sym_to_val) if ub is not None else None
                _assume_bounds_fn(val, lb=lb_val, ub=ub_val)

            # Emit layout
            layout_val = emit_layout_from_python(layout, sym_to_val)

            # Emit apply or apply_inverse
            if mode == 'apply':
                arg_vals = [_resolve_dim(a, sym_to_val) for a in args]
                result = ApplyOp(flat_index=idx_ty, layout=layout_val,
                                 indices=arg_vals).result
                _func_dialect.ReturnOp([result])
            else:
                flat_val = _resolve_dim(args, sym_to_val)
                inv_op = ApplyInverseOp(
                    indices=[idx_ty] * rank,
                    layout=layout_val,
                    flat_index=flat_val,
                )
                _func_dialect.ReturnOp(list(inv_op.results))

        # Run lego-lower pipeline
        if _LEGO_DEBUG:
            print("=== MLIR input (LEGO dialect) ===", file=sys.stderr)
            print(module, file=sys.stderr)
            print(file=sys.stderr)

            # Show intermediate: after canonicalize + CSE only (no LEGO passes)
            module_copy = Module.parse(str(module))
            pm_pre = _PassManager.parse(
                "builtin.module(canonicalize,cse)"
            )
            pm_pre.run(module_copy.operation)
            print("=== MLIR after canonicalize + CSE ===", file=sys.stderr)
            print(module_copy, file=sys.stderr)
            print(file=sys.stderr)

        pm = _PassManager.parse("builtin.module(lego-lower)")
        pm.run(module.operation)

        if _LEGO_DEBUG:
            print("=== MLIR after lego-lower (fully simplified) ===", file=sys.stderr)
            print(module, file=sys.stderr)
            print(file=sys.stderr)

        # Extract the function after lowering (passes may rebuild ops)
        func_op = None
        for op in module.body:
            if op.name == "roundtrip" or op.name == "func.func":
                func_op = op
                break
        if func_op is None:
            # Fallback: grab first op in module body
            for op in module.body:
                func_op = op
                break

        # Rebuild val_to_sym using entry block arguments by index
        entry_block = func_op.regions[0].blocks[0]
        val_to_sym_post = {}
        for i, sym in enumerate(sym_list):
            val_to_sym_post[entry_block.arguments[i]] = sym

        # Find the return op
        return_op = None
        for op in entry_block:
            if op.name == "func.return":
                return_op = op
                break

        # Convert results back to SymPy
        results_sympy = []
        for operand in return_op.operands:
            results_sympy.append(arith_to_sympy(operand, val_to_sym_post))

        if mode == 'apply':
            return results_sympy[0]
        return results_sympy
