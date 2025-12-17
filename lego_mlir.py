from lego import *
import functools

from mlir.dialects import arith, scf, func, affine, gpu, memref, vector
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

    @staticmethod
    def get_constant_int(val):
        return arith.ConstantOp(T.i32(), val).result

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
                token = gpu.wait(token_ty, [])
                # allocate input
                for i in set(ins + outs):
                    token = i.gpu_allocate(token)
                    i.host_allocate()

                for i in ins:
                    i.fill_host()
                    token = i.copy_to_device(token)

                gpu.WaitOp(token_ty, [token])

                launch_op = gpu.LaunchOp(
                    None,
                    [],
                    *map(arith.ConstantOp.create_index, gridSize),
                    *map(arith.ConstantOp.create_index, blockSize)
                )
                launch_op.attributes["workgroup_attributions"] = IntegerAttr.get(
                    T.i64(), len(workgroup_memory))
                launch_op.body.blocks.append(
                    *([T.index()] * 12 + [w.get_memref_type_address_space(3) for w in workgroup_memory] + [p.get_memref_type_address_space(5) for p in private_memory]))
                with InsertionPoint(launch_op.body.blocks[0]):
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

    # @staticmethod
    # def contract(smem_a: MatrixLayout, smem_b: MatrixLayout, smem_c: MatrixLayout, size=[16, 8, 16], input=[[0, 2], [2, 1]], output=[0, 1], type=None):
    #     def get_elements_by_indices(lst, indices):
    #         return [lst[i] for i in indices]
    #     vector_output = VectorType.get(
    #         get_elements_by_indices(size, output), type)
    #     vector_input0 = VectorType.get(
    #         get_elements_by_indices(size, input[0]),  type)
    #     vector_input1 = VectorType.get(
    #         get_elements_by_indices(size, input[1]),  type)
    #     index = IndexType.get()

    #     # Create constants
    #     cst_0 = arith.ConstantOp(vector_output, DenseElementsAttr.get_splat(
    #         vector_output, FloatAttr.get(type, 0.0)))

    #     # Identity permutation map for default
    #     identity_map = AffineMap.get_identity(2)
    #     for i in [x * 16 for x in range(2)]:
    #         # Create vector.transfer_read for %A
    #         c0 = arith.ConstantOp(index, IntegerAttr.get(index, i))
    #         c3 = arith.ConstantOp(index, IntegerAttr.get(index, 0))
    #         cst = arith.ConstantOp(type, FloatAttr.get(type, 0.0))
    #         A = vector.TransferReadOp(
    #             vector_input0,
    #             smem_a.get_memory_ref_address_space(),
    #             [c0.result, c3.result],
    #             permutation_map=identity_map,
    #             padding=cst.result,
    #             in_bounds=[True, True],
    #         )
    #         for j in [x * 8 for x in range(4)]:
    #             c1 = arith.ConstantOp(index, IntegerAttr.get(index, j))

    #             # Create permutation map for %B (as per 'permutation_map = #map0' in original MLIR code)
    #             # Define map0; in this case, we'll use the identity map
    #             map0 = AffineMap.get_identity(2)

    #             # Create vector.transfer_read for %B
    #             B = vector.TransferReadOp(
    #                 vector_input1,
    #                 smem_b.get_memory_ref_address_space(),
    #                 [c3.result, c1.result],
    #                 permutation_map=map0,
    #                 padding=cst.result,
    #                 in_bounds=[True, True],
    #             )

    #             # Define affine dimension expressions for indexing maps
    #             ds = [AffineDimExpr.get(i) for i in range(len(size))]

    #             # Define indexing maps for vector.contract
    #             # TODO check here for the right thing ik * kj -> ij
    #             map1 = AffineMap.get(
    #                 # For A
    #                 len(size), 0, get_elements_by_indices(ds, input[0]))
    #             map2 = AffineMap.get(
    #                 # For B
    #                 len(size), 0, get_elements_by_indices(ds, input[1]))
    #             # For D
    #             map3 = AffineMap.get(
    #                 len(size), 0, get_elements_by_indices(ds, output))

    #             # Create vector.contract operation
    #             D = vector.contract(
    #                 vector_output,
    #                 lhs=A.result,
    #                 rhs=B.result,
    #                 acc=cst_0.result,
    #                 indexing_maps=ArrayAttr.get(
    #                     [AffineMapAttr.get(am) for am in [map1, map2, map3]]
    #                 ),
    #                 iterator_types=ArrayAttr.get([Attribute.parse(str(am)) for am in [
    #                     "#vector.iterator_type<parallel>", "#vector.iterator_type<parallel>", "#vector.iterator_type<reduction>"]]),
    #                 kind="add"
    #             )

    #             vector.transfer_write(
    #                 None,
    #                 D,
    #                 smem_c.get_memory_ref_address_space(),
    #                 [c0.result, c1.result],
    #                 map0,
    #                 in_bounds=[True, True],
    #             )


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

    def dealloc_host(self):
        memref.dealloc(self.alloc_ref)
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

    def print_matrix(self, orderBy, M, N):
        self.layout = orderBy.TileBy([M, N])
        for i in affine.for_(0, M, step=1):
            for j in affine.for_(0, N, step=1):
                ii = printer.from_ssa_to_sym(printer.get_int_from_index(i))
                jj = printer.from_ssa_to_sym(printer.get_int_from_index(j))

                value_read = self[ii, jj]
                gpu.printf("%.0f\t", [value_read])
                affine.yield_([])
            gpu.printf("\n", [])
            affine.yield_([])
        gpu.printf("\n", [])

    def print_matrix_kernel(self, mem, M, N):
        launch_op = gpu.LaunchOp(
            None,
            [],
            *map(arith.ConstantOp.create_index, [1, 1, 1]),
            *map(arith.ConstantOp.create_index, [1, 1, 1])
        )
        launch_op.body.blocks.append(*([T.index()] * 12))
        with ir.InsertionPoint(launch_op.body.blocks[0]):
            self.print_matrix(mem, M, N)
            gpu.terminator()

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
