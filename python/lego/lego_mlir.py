from .lego import *
import functools

from mlir.dialects import arith, scf, func, memref
try:
    from mlir.dialects import affine, gpu
except ImportError:
    affine = None
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


class SympyMLIRPrinter:

    def __init__(self, ctx=Context(), int_width=32):
        # 1) Create a context and register any dialects you need.
        self.ctx = ctx
        self.ctx.allow_unregistered_dialects = True

        # 2) Pre-build your integer type.
        self.int_type = IntegerType.get_signless(int_width, self.ctx)
        self.bool_type = IntegerType.get_signless(1, self.ctx)

        # top-level MLIR module
        with self.ctx, Location.unknown(self.ctx):
            self.module = Module.create()

        # symbol→Value map
        self.sym_map = {}
        self.sym_i = 0

    def get_unique_symbol_name(self):
        self.sym_i += 1
        return sp.Symbol(f"tmp_{self.sym_i}")

    @staticmethod
    def get_int_from_index(val):
        return arith.index_cast(T.i32(), val)

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
                        # If a schedule is provided, parse and insert it
                    if schedule:
                        # Parse the transform module
                        transform_module = Module.parse(schedule)
                        print(transform_module)
                    print(module)
            return wrapper()
        return decorator_body

    def from_ssa_to_sym(self, val):
        # if the value is present in the map find the key and return the key
        for key, value in self.sym_map.items():
            if value == val:
                return key
        val_sym = self.get_unique_symbol_name()
        self.sym_map[val_sym] = val
        return val_sym

    def insert_barrier(self):
        gpu.barrier()


    def generate_loop(self, l: 'GroupBy', step=1, iter_args=[]):
        def decorator_body(body):
            @functools.wraps(body)
            def wrapper(*args, **kwargs):
                for_op = affine.AffineForOp(
                    0, product(l.dims()), step, iter_args=iter_args)
                idx = for_op.induction_variable
                with InsertionPoint(for_op.body):
                    idx = self.get_int_from_index(idx)
                    idx_sym = self.from_ssa_to_sym(idx)
                    pargs = l.inv(idx_sym)
                    if len(iter_args) > 0:
                        res = body(pargs, idx, tuple(
                            for_op.inner_iter_args))
                        affine.yield_(res)
                    else:
                        body(pargs, idx)
                        affine.yield_([])
                if len(iter_args) > 0:
                    return for_op.results
            if len(iter_args) > 0:
                return wrapper
            return wrapper()
        return decorator_body

    @staticmethod
    def get_token_type():
        return ir.Type.parse("!gpu.async.token")

    @staticmethod
    def generate_gpu_kernel(ins: List['MLIRTensor'], outs: List['MLIRTensor'], gridSize, blockSize, workgroup_memory=[], private_memory=[]):
        def decorator_body(body):
            @functools.wraps(body)
            def wrapper():
                token_ty = SympyMLIRPrinter.get_token_type()
                token = gpu.wait([])
                # allocate input
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
                
                # Use the existing block created by LaunchOp
                block = launch_op.body.blocks[0]
                # The block already has 12 arguments (indices). We append workgroup and private memory arguments.
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
                        workgroup_memory[i].set_memory_space(
                            MemorySpace.SHARED_MEMORY)
                    for i in private_memory:
                        i.set_memory_space(MemorySpace.PRIVATE_MEMORY)
                    body(launch_op.body.blocks[0].arguments)
                    gpu.terminator()

                for i in set(ins + outs):
                    token = i.dealloc_gpu(token)
            return wrapper()
        return decorator_body

    def get_block_linear_index(self,):
        return self.from_ssa_to_sym(arith.index_cast(T.i32(), gpu.block_id(gpu.Dimension.x)))

    def get_thread_linear_index(self, ):
        return self.from_ssa_to_sym(arith.index_cast(T.i32(), gpu.thread_id(gpu.Dimension.x)))

    def translate(self, expr):
        # 4) Every time we lower, re-enter the context, location, and insertion point.
        # with self.ctx:
        #     with Location.unknown(self.ctx):
        #         with InsertionPoint(self.module.body):
        return self._lower(expr)

    def _lower(self, expr):
        if isinstance(expr, sp.Symbol):
            if expr not in self.sym_map:
                raise KeyError(f"No MLIR SSA value for symbol {expr}")
            return self.sym_map[expr]
        # 0) Relational → arith.cmpi producing an i1
        if isinstance(expr, sp.core.relational.Relational):
            lhs_v = self._lower(expr.lhs)
            rhs_v = self._lower(expr.rhs)
            # pick the unsigned predicate (UGT, ULE, etc.)
            if isinstance(expr, sp.StrictGreaterThan):
                pred = arith.CmpIPredicate.ugt
            elif isinstance(expr, sp.GreaterThan):
                pred = arith.CmpIPredicate.uge
            elif isinstance(expr, sp.StrictLessThan):
                pred = arith.CmpIPredicate.ult
            elif isinstance(expr, sp.LessThan):
                pred = arith.CmpIPredicate.ule
            elif isinstance(expr, sp.Equality):
                pred = arith.CmpIPredicate.eq
            elif isinstance(expr, sp.Unequality):
                pred = arith.CmpIPredicate.ne
            else:
                raise NotImplementedError(f"Unsupported relational: {expr}")
            op = arith.CmpIOp(pred, lhs_v, rhs_v)
            return op.result

        # Integer constants
        if isinstance(expr, sp.Integer) or isinstance(expr, int):
            val = int(expr)
            op = arith.ConstantOp(self.int_type, val)
            return op.result

        # Add -> nested addui.extended ops
        if isinstance(expr, sp.Add):
            results = [self._lower(a) for a in expr.args]
            acc = results[0]
            for r in results[1:]:
                op = arith.addi(acc, r)
                acc = op
            return acc

        # Mul -> muli (works for unsigned)
        if isinstance(expr, sp.Mul):
            results = [self._lower(a) for a in expr.args]
            acc = results[0]
            for r in results[1:]:
                op = arith.muli(acc, r)
                acc = op
            return acc

        # FloorDiv or Div -> unsigned div
        if isinstance(expr, sp.floor):
            inner = expr.args[0]              # this is “x/y”
            num, den = inner.as_numer_denom()  # split into numerator & denominator
            a = self._lower(num)
            b = self._lower(den)
            op = arith.divui(a, b)
            return op

        if getattr(expr, 'is_Div', False):
            a, b = (self._lower(a) for a in expr.args)
            op = arith.divui(a, b)
            return op

        # Mod -> unsigned remainder
        if isinstance(expr, sp.Mod):
            a, b = (self._lower(a) for a in expr.args)
            op = arith.remui(a, b)
            return op

        # Piecewise -> nested scf.if chains
        if isinstance(expr, sp.Piecewise):
            return self._lower_piecewise(expr)

        raise NotImplementedError(f"Unsupported Sympy node: {expr}")

    def _lower_piecewise(self, pw):
        def recurse(i):
            val, cond = pw.args[i]
            then_val = self._lower(val)
            # last clause is the “else”
            if i == len(pw.args) - 1:
                return then_val
            cond_val = self._lower(cond)
            ifop = scf.IfOp(cond_val, [self.int_type], has_else_region=True)
            with InsertionPoint(ifop.then_block):
                scf.YieldOp(then_val)
            with InsertionPoint(ifop.else_block):
                scf.YieldOp(recurse(i+1))
            return ifop.results[0]

        return recurse(0)

class MLIRTensor:
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
        physical_shape = self.layout.objects[0].dims()
        if self.is_dim_shape:
            return T.memref(*physical_shape, self.data_type, memory_space=Attribute.parse("#gpu.address_space<workgroup>") if address_space == 3 else address_space)
        return T.memref(self.get_flattend_shape(), self.data_type, memory_space=address_space)

    def host_allocate(self):
        self.alloc_ref = memref.alloc(
            self.get_memref_type_address_space(0), [], [])
        return self

    def dealloc_gpu(self, *tokens):
        if tokens is None:
            gpu.dealloc(self.gpu_alloc_ref)
            return None
        else:
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
        else:
            token_ty = Type.parse("!gpu.async.token")
            tmp = gpu.alloc(
                self.get_memref_type_address_space(0), token_ty, list(tokens), [], [])
            self.gpu_alloc_ref = tmp[0]
        return tmp[1]

    def fill_host(self):
        for i in affine.for_(0, self.get_flattend_shape(), step=1):
            f_i = arith.sitofp(self.data_type, arith.index_cast(T.i32(), i))
            self.store_physical_1d([i], f_i)
            affine.yield_([])
        return self

    def store_physical_1d(self, coords, value):
        return memref.store(value, self.get_memory_ref_address_space(), coords)

    def copy_to_device(self, token):
        token_ty = Type.parse("!gpu.async.token")
        if token is None:
            gpu.memcpy(None, [], self.gpu_alloc_ref, self.alloc_ref)
            return None
        else:
            return gpu.memcpy(token_ty, [token], self.gpu_alloc_ref, self.alloc_ref)

    def get_memory_ref_address_space(self):
        memory_space = self.memory_space
        if memory_space == MemorySpace.SHARED_MEMORY:
            # return memref.get_global(self.get_memref_type_address_space(3), self.shared_memory_symbol)
            return self.shared_memory_ref
        if memory_space == MemorySpace.PRIVATE_MEMORY:
            # return memref.get_global(self.get_memref_type_address_space(3), self.shared_memory_symbol)
            return self.private_memory_ref
        if memory_space == MemorySpace.GLOBAL_MEMORY:
            return self.gpu_alloc_ref
        if memory_space == MemorySpace.HOST_MEMORY:
            return self.alloc_ref
        return self.alloc_ref

    def __getitem__(self, key):
        dummy_to_tr = {}
        result = []
        i = 0
        for x in key:
            sym = sp.symbols(f"_tr{i}", integer=True)
            dummy_to_tr[sym] = x
            i += 1
            result.append(sym)
        apply_map = self.layout[tuple(result)]
        apply_map = apply_map.xreplace(dummy_to_tr)
        apply_map = [arith.index_cast(T.index(), printer.translate(apply_map))]
        return memref.load(self.get_memory_ref_address_space(), apply_map)

    def __setitem__(self, key, value):
        dummy_to_tr = {}
        result = []
        i = 0
        for x in key:
            sym = sp.symbols(f"_tr{i}", integer=True)
            dummy_to_tr[sym] = x
            i += 1
            result.append(sym)
        apply_map = self.layout[tuple(result)]
        apply_map = apply_map.xreplace(dummy_to_tr)
        apply_map = [arith.index_cast(
            T.index(),  printer.translate(apply_map))]
        return memref.store(value, self.get_memory_ref_address_space(), apply_map)


printer = SympyMLIRPrinter()


# ============================================================================
# MLIR Roundtrip: replace simplify_ops (SymPy+z3) with MLIR pipeline
# ============================================================================

from mlir.ir import IndexType, FunctionType, StringAttr, UnitAttr
from mlir.passmanager import PassManager
from mlir.dialects import func as func_dialect
from lego.dialects.lego_dialect import (
    register as register_lego,
    RegPOp, RowOp, ColOp, OrderByOp, GroupByOp, ApplyOp,
    ApplyInverseOp, TileByOp, GenPOp, YieldOp,
)
from lego.dialects._lego_ops_gen import assume_bounds as _assume_bounds


def _index_const(val):
    """Emit arith.constant with index type."""
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
        # Separate integer coefficient from rest
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
            # x^-1 shouldn't appear at index level; skip
            raise NotImplementedError(f"Negative power in index expr: {expr}")
        if exp.is_Integer and int(exp) >= 0:
            b = _lower_sympy_to_index(base, sym_to_val)
            result = _index_const(1)
            for _ in range(int(exp)):
                result = arith.muli(result, b)
            return result

    raise NotImplementedError(f"Cannot lower SymPy expr to index: {expr}")


def emit_layout_from_python(layout, sym_to_val):
    """Convert a Python layout object to MLIR LEGO dialect ops.

    Args:
        layout: Python LayoutBlock (RegP, OrderBy, GroupBy, GenP)
        sym_to_val: dict {sp.Symbol | int -> MLIR Value}
    Returns:
        MLIR Value of !lego.layout type
    """
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
            # Create temp SymPy symbols and call the Python f_apply
            temp_syms = [sp.Symbol(f"_genp_arg_{k}", integer=True) for k in range(rank)]
            local_map = dict(sym_to_val)
            for s, arg in zip(temp_syms, apply_block.arguments):
                local_map[s] = arg
            sympy_result = layout.f_apply(tuple(temp_syms))
            mlir_result = _lower_sympy_to_index(sympy_result, local_map)
            YieldOp(values=[mlir_result])

        # Inverse region: block arg = single flat index -> yield N indices
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
        val = int(ir.IntegerAttr(attr).value)
        result = sp.Integer(val)
        memo[value] = result
        return result

    if op_name == "arith.addi":
        a = arith_to_sympy(op.operands[0], val_to_sym, memo)
        b = arith_to_sympy(op.operands[1], val_to_sym, memo)
        result = a + b
        memo[value] = result
        return result

    if op_name == "arith.muli":
        a = arith_to_sympy(op.operands[0], val_to_sym, memo)
        b = arith_to_sympy(op.operands[1], val_to_sym, memo)
        result = a * b
        memo[value] = result
        return result

    if op_name == "arith.subi":
        a = arith_to_sympy(op.operands[0], val_to_sym, memo)
        b = arith_to_sympy(op.operands[1], val_to_sym, memo)
        result = a - b
        memo[value] = result
        return result

    if op_name == "arith.divui":
        a = arith_to_sympy(op.operands[0], val_to_sym, memo)
        b = arith_to_sympy(op.operands[1], val_to_sym, memo)
        result = sp.floor(a / b)
        memo[value] = result
        return result

    if op_name == "arith.remui":
        a = arith_to_sympy(op.operands[0], val_to_sym, memo)
        b = arith_to_sympy(op.operands[1], val_to_sym, memo)
        result = sp.Mod(a, b)
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
    if isinstance(layout, OrderBy):
        for p in layout.perms:
            syms |= _collect_free_symbols(p)
    elif isinstance(layout, GroupBy):
        for obj in layout.objects:
            syms |= _collect_free_symbols(obj)
    return syms


def simplify_via_mlir(layout, mode, args, constraints=None):
    """Compute layout.apply or layout.inv via MLIR roundtrip.

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
    register_lego(ctx)

    with ctx, Location.unknown():
        module = Module.create()
        idx_ty = IndexType.get()

        # Determine function signature
        n_args = len(sym_list)
        if mode == 'apply':
            rank = len(layout._dims)
            func_ty = FunctionType.get([idx_ty] * n_args, [idx_ty])
        else:
            # inv returns rank indices
            rank = len(layout._dims)
            func_ty = FunctionType.get([idx_ty] * n_args, [idx_ty] * rank)

        with InsertionPoint(module.body):
            f = func_dialect.FuncOp("roundtrip", func_ty)
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
                _assume_bounds(val, lb=lb_val, ub=ub_val)

            # Emit layout
            layout_val = emit_layout_from_python(layout, sym_to_val)

            # Emit apply or apply_inverse
            if mode == 'apply':
                arg_vals = [_resolve_dim(a, sym_to_val) for a in args]
                result = ApplyOp(flat_index=idx_ty, layout=layout_val,
                                 indices=arg_vals).result
                func_dialect.ReturnOp([result])
            else:
                flat_val = _resolve_dim(args, sym_to_val)
                inv_op = ApplyInverseOp(
                    indices=[idx_ty] * rank,
                    layout=layout_val,
                    flat_index=flat_val,
                )
                func_dialect.ReturnOp(list(inv_op.results))

        # Run lego-lower pipeline
        pm = PassManager.parse("builtin.module(lego-lower)")
        pm.run(module.operation)

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
