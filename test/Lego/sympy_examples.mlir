// RUN: lego-opt %s \
// RUN:   --lego-materialize-assume-bounds \
// RUN:   -lego-lower \
// RUN:   --lego-arith-simplification --int-range-optimizations --canonicalize --cse \
// RUN:   --lego-arith-simplification --int-range-optimizations --canonicalize --cse \
// RUN:   --lego-arith-simplification --int-range-optimizations --canonicalize --cse \
// RUN:   --lego-arith-simplification --int-range-optimizations --canonicalize --cse \
// RUN:   -split-input-file | FileCheck %s
//
// ============================================================================
// Tests derived from Python/SymPy layout examples
//
// Validates that MLIR layout lowering produces the same indexing as the
// corresponding SymPy expressions from:
//   python/examples/graphene.py
//   paper/benchmarks/cuda/bricks_f3d.py     (normal, bricks, const)
//   paper/benchmarks/cuda/bricks_laplasian.py (same layouts)
//   paper/benchmarks/cuda/lud.py            (GroupBy for thread coarsening)
//
// Dimensions: N=8, B=2  (scaled down from N=384, B=8 for test brevity)
//
// SymPy-verified expressions (run via python/lego/lego.py):
//   graphene:   i + 8*j + 2*k + 16*q + 4*w
//   const_out:  64*a + 8*b + c
//   lud_fwd:    32*ii + 16*jj + tidx + 4*tidy
//   normal_out: (bx*2+i)*64 + (by*2+j)*8 + (bz*2+k)  [modular arith]
//   bricks_out: (bx*16+by*4+bz)*8 + (i*4+j*2+k)      [modular arith]
// ============================================================================

// -----

// ============================================================================
// const = OrderBy(Row(8, 8, 8)).TileBy([8, 8, 8])
// SymPy: const[a, b, c] = 64*a + 8*b + c
// ============================================================================

// CHECK-LABEL: func.func @const_3d_apply
// CHECK-SAME:  (%[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index)
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[T1:.*]] = arith.muli %[[B]], %[[C8]] : index
// CHECK:       %[[T2:.*]] = arith.addi %[[C]], %[[T1]] : index
// CHECK:       %[[T3:.*]] = arith.muli %[[A]], %[[C64]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[T2]], %[[T3]] : index
// CHECK:       return %[[RES]] : index
func.func @const_3d_apply(%a: index, %b: index, %c: index) -> index {
  %c8 = arith.constant 8 : index
  %row = lego.row [%c8, %c8, %c8] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[%c8, %c8, %c8]] : !lego.layout
  %f = lego.apply %tiled(%a, %b, %c) : !lego.layout
  return %f : index
}

// -----

// ============================================================================
// graphene.py: OrderBy(RegP([2,2,2,2,2], [4,1,3,2,0])).GroupBy([(2,2,2,2,2)])
// SymPy: l[i, j, k, w, q] = i + 8*j + 2*k + 16*q + 4*w
//
// With assume_bounds, the flatten→unflatten chain simplifies via
// lego-materialize-assume-bounds (remui) + A2 + int-range folding
// to the direct permuted expression.
// ============================================================================

// CHECK-LABEL: func.func @graphene_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index, %[[W:.*]]: index, %[[Q:.*]]: index)
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[IB:.*]] = arith.remui %[[I]], %[[C2]] : index
// CHECK:       %[[JB:.*]] = arith.remui %[[J]], %[[C2]] : index
// CHECK:       %[[KB:.*]] = arith.remui %[[K]], %[[C2]] : index
// CHECK:       %[[WB:.*]] = arith.remui %[[W]], %[[C2]] : index
// CHECK:       %[[QB:.*]] = arith.remui %[[Q]], %[[C2]] : index
// CHECK:       %[[MK:.*]] = arith.muli %[[KB]], %[[C2]] : index
// CHECK:       %[[S1:.*]] = arith.addi %[[IB]], %[[MK]] : index
// CHECK:       %[[MW:.*]] = arith.muli %[[WB]], %[[C4]] : index
// CHECK:       %[[S2:.*]] = arith.addi %[[S1]], %[[MW]] : index
// CHECK:       %[[MJ:.*]] = arith.muli %[[JB]], %[[C8]] : index
// CHECK:       %[[S3:.*]] = arith.addi %[[S2]], %[[MJ]] : index
// CHECK:       %[[MQ:.*]] = arith.muli %[[QB]], %[[C16]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[S3]], %[[MQ]] : index
// CHECK:       return %[[RES]] : index
func.func @graphene_apply(%i: index, %j: index, %k: index, %w: index, %q: index) -> index {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  lego.assume_bounds %i lb: %c0 ub: %c2
  lego.assume_bounds %j lb: %c0 ub: %c2
  lego.assume_bounds %k lb: %c0 ub: %c2
  lego.assume_bounds %w lb: %c0 ub: %c2
  lego.assume_bounds %q lb: %c0 ub: %c2
  %regp = lego.reg_p perm [4, 1, 3, 2, 0] dims [%c2, %c2, %c2, %c2, %c2] : !lego.layout
  %ob = lego.order_by(%regp) : !lego.layout
  %gb = lego.group_by [%c2, %c2, %c2, %c2, %c2](%ob) : !lego.layout
  %f = lego.apply %gb(%i, %j, %k, %w, %q) : !lego.layout
  return %f : index
}

// -----

// ============================================================================
// normal = OrderBy(Row(N, N, N)).TileBy([N//B, N//B, N//B], [B, B, B])
//   with N=8, B=2  =>  OrderBy(Row(8,8,8)).TileBy([4,4,4], [2,2,2])
// SymPy: normal[bx, by, bz, i, j, k] = (bx*2+i)*64 + (by*2+j)*8 + (bz*2+k)
//
// TileBy normalizes to GroupBy with interleave permutation.
// The lowered form retains divui/remui (modular arithmetic equivalent).
// ============================================================================

// CHECK-LABEL: func.func @normal_3d_apply
// CHECK-SAME:  (%[[BX:.*]]: index, %[[BY:.*]]: index, %[[BZ:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       arith.remui %[[BX]], %[[C4]] : index
// CHECK:       arith.remui %[[BY]], %[[C4]] : index
// CHECK:       arith.remui %[[BZ]], %[[C4]] : index
// CHECK:       arith.remui %[[I]], %[[C2]] : index
// CHECK:       arith.remui %[[J]], %[[C2]] : index
// CHECK:       arith.remui %[[K]], %[[C2]] : index
// CHECK:       arith.muli
// CHECK:       arith.addi
// CHECK:       arith.divui
// CHECK:       arith.remui
// CHECK:       arith.muli {{.*}}, %[[C128]] : index
// CHECK:       %[[RES:.*]] = arith.addi
// CHECK:       return %[[RES]] : index
func.func @normal_3d_apply(%bx: index, %by: index, %bz: index,
                           %i: index, %j: index, %k: index) -> index {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  lego.assume_bounds %bx lb: %c0 ub: %c4
  lego.assume_bounds %by lb: %c0 ub: %c4
  lego.assume_bounds %bz lb: %c0 ub: %c4
  lego.assume_bounds %i lb: %c0 ub: %c2
  lego.assume_bounds %j lb: %c0 ub: %c2
  lego.assume_bounds %k lb: %c0 ub: %c2
  %row = lego.row [%c8, %c8, %c8] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[%c4, %c4, %c4], [%c2, %c2, %c2]] : !lego.layout
  %f = lego.apply %tiled(%bx, %by, %bz, %i, %j, %k) : !lego.layout
  return %f : index
}

// -----

// ============================================================================
// bricks = OrderBy(Row(N//B, N//B, N//B), Row(B, B, B))
//            .TileBy([N//B, N//B, N//B], [B, B, B])
//   with N=8, B=2  =>  OrderBy(Row(4,4,4), Row(2,2,2)).TileBy([4,4,4],[2,2,2])
// SymPy: bricks[bx, by, bz, i, j, k]
//      = (bx*16 + by*4 + bz) * 8 + (i*4 + j*2 + k)
//
// TileBy normalizes to GroupBy with interleave permutation.
// The lowered form retains divui/remui (modular arithmetic equivalent).
// ============================================================================

// CHECK-LABEL: func.func @bricks_3d_apply
// CHECK-SAME:  (%[[BX:.*]]: index, %[[BY:.*]]: index, %[[BZ:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       arith.remui %[[BX]], %[[C4]] : index
// CHECK:       arith.remui %[[BY]], %[[C4]] : index
// CHECK:       arith.remui %[[BZ]], %[[C4]] : index
// CHECK:       arith.remui %[[I]], %[[C2]] : index
// CHECK:       arith.remui %[[J]], %[[C2]] : index
// CHECK:       arith.remui %[[K]], %[[C2]] : index
// CHECK:       arith.muli
// CHECK:       arith.addi
// CHECK:       arith.divui
// CHECK:       arith.remui
// CHECK:       arith.muli {{.*}}, %[[C8]] : index
// CHECK:       arith.addi
// CHECK:       return {{.*}} : index
func.func @bricks_3d_apply(%bx: index, %by: index, %bz: index,
                           %i: index, %j: index, %k: index) -> index {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  lego.assume_bounds %bx lb: %c0 ub: %c4
  lego.assume_bounds %by lb: %c0 ub: %c4
  lego.assume_bounds %bz lb: %c0 ub: %c4
  lego.assume_bounds %i lb: %c0 ub: %c2
  lego.assume_bounds %j lb: %c0 ub: %c2
  lego.assume_bounds %k lb: %c0 ub: %c2
  %row1 = lego.row [%c4, %c4, %c4] : !lego.layout
  %row2 = lego.row [%c2, %c2, %c2] : !lego.layout
  %ob = lego.order_by(%row1, %row2) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[%c4, %c4, %c4], [%c2, %c2, %c2]] : !lego.layout
  %f = lego.apply %tiled(%bx, %by, %bz, %i, %j, %k) : !lego.layout
  return %f : index
}

// -----

// ============================================================================
// lud.py: OrderBy(Row(R*T, R*T)).GroupBy([(R, R), (T, T)])
//   with R=2, T=4  =>  OrderBy(Row(8, 8)).GroupBy([(2, 2), (4, 4)])
// SymPy: l[ii, jj, tidy, tidx] = 32*ii + 16*jj + tidx + 4*tidy
// ============================================================================

// CHECK-LABEL: func.func @lud_groupby_apply
// CHECK-SAME:  (%[[II:.*]]: index, %[[JJ:.*]]: index, %[[TIDY:.*]]: index, %[[TIDX:.*]]: index)
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[T1:.*]] = arith.muli %[[TIDY]], %[[C4]] : index
// CHECK:       %[[S1:.*]] = arith.addi %[[TIDX]], %[[T1]] : index
// CHECK:       %[[T2:.*]] = arith.muli %[[JJ]], %[[C16]] : index
// CHECK:       %[[S2:.*]] = arith.addi %[[S1]], %[[T2]] : index
// CHECK:       %[[T3:.*]] = arith.muli %[[II]], %[[C32]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[S2]], %[[T3]] : index
// CHECK:       return %[[RES]] : index
func.func @lud_groupby_apply(%ii: index, %jj: index,
                             %tidy: index, %tidx: index) -> index {
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %row = lego.row [%c8, %c8] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %gb = lego.group_by [%c2, %c2, %c4, %c4](%ob) : !lego.layout
  %f = lego.apply %gb(%ii, %jj, %tidy, %tidx) : !lego.layout
  return %f : index
}

// -----

// ============================================================================
// lud.py inverse: l.inv(flat) -> (ii, jj, tidy, tidx)
// SymPy: l.inv((ii*R+jj)*T*T + tid) = [ii, jj, tid/T, tid%T]
//   with R=2, T=4:
//     ii   = flat / 32
//     jj   = (flat / 16) % 2
//     tidy = (flat / 4) % 4
//     tidx = flat % 4
// ============================================================================

// CHECK-LABEL: func.func @lud_groupby_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK:       %[[R64:.*]] = arith.remui %[[F]], %[[C64]] : index
// CHECK:       %[[II:.*]] = arith.divui %[[R64]], %[[C32]] : index
// CHECK:       %[[R32:.*]] = arith.remui %[[R64]], %[[C32]] : index
// CHECK:       %[[JJ:.*]] = arith.divui %[[R32]], %[[C16]] : index
// CHECK:       %[[R16:.*]] = arith.remui %[[R32]], %[[C16]] : index
// CHECK:       %[[TIDY:.*]] = arith.divui %[[R16]], %[[C4]] : index
// CHECK:       %[[TIDX:.*]] = arith.remui %[[R16]], %[[C4]] : index
// CHECK:       return %[[II]], %[[JJ]], %[[TIDY]], %[[TIDX]] : index, index, index, index
func.func @lud_groupby_inv(%f: index) -> (index, index, index, index) {
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %row = lego.row [%c8, %c8] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %gb = lego.group_by [%c2, %c2, %c4, %c4](%ob) : !lego.layout
  %ii, %jj, %tidy, %tidx = lego.apply_inverse %gb(%f) : !lego.layout -> index, index, index, index
  return %ii, %jj, %tidy, %tidx : index, index, index, index
}

// -----

// ============================================================================
// Stencil indexing with offsets (from bricks_f3d.py / bricks_laplasian.py)
// ============================================================================

// CHECK-LABEL: func.func @normal_stencil_load
// CHECK-SAME:  (%[[MEM:.*]]: memref<512xf32>,
// CHECK:       arith.addi
// CHECK:       arith.muli
// CHECK:       %[[VAL:.*]] = memref.load %[[MEM]][%{{.*}}]
// CHECK:       return %[[VAL]] : f32
func.func @normal_stencil_load(%mem: memref<512xf32>,
    %bx: index, %by: index, %bz: index,
    %i: index, %j: index, %k: index,
    %i_diff: index, %j_diff: index, %k_diff: index) -> f32 {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %row = lego.row [%c8, %c8, %c8] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[%c4, %c4, %c4], [%c2, %c2, %c2]] : !lego.layout
  %view = lego.cast_view %mem, %tiled : memref<512xf32>, !lego.layout -> !lego.view<f32>
  %io = arith.addi %i, %i_diff : index
  %jo = arith.addi %j, %j_diff : index
  %ko = arith.addi %k, %k_diff : index
  %val = lego.load %view[%bx, %by, %bz, %io, %jo, %ko] : !lego.view<f32>, index, index, index, index, index, index
  return %val : f32
}

// -----

// CHECK-LABEL: func.func @bricks_stencil_load
// CHECK-SAME:  (%[[MEM:.*]]: memref<512xf32>,
// CHECK:       arith.addi
// CHECK:       arith.muli
// CHECK:       %[[VAL:.*]] = memref.load %[[MEM]][%{{.*}}]
// CHECK:       return %[[VAL]] : f32
func.func @bricks_stencil_load(%mem: memref<512xf32>,
    %bx: index, %by: index, %bz: index,
    %i: index, %j: index, %k: index,
    %i_diff: index, %j_diff: index, %k_diff: index) -> f32 {
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %row1 = lego.row [%c4, %c4, %c4] : !lego.layout
  %row2 = lego.row [%c2, %c2, %c2] : !lego.layout
  %ob = lego.order_by(%row1, %row2) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[%c4, %c4, %c4], [%c2, %c2, %c2]] : !lego.layout
  %view = lego.cast_view %mem, %tiled : memref<512xf32>, !lego.layout -> !lego.view<f32>
  %io = arith.addi %i, %i_diff : index
  %jo = arith.addi %j, %j_diff : index
  %ko = arith.addi %k, %k_diff : index
  %val = lego.load %view[%bx, %by, %bz, %io, %jo, %ko] : !lego.view<f32>, index, index, index, index, index, index
  return %val : f32
}

// -----

// CHECK-LABEL: func.func @const_stencil_load
// CHECK-SAME:  (%[[MEM:.*]]: memref<512xf32>,
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       arith.addi
// CHECK:       arith.muli
// CHECK:       %[[VAL:.*]] = memref.load %[[MEM]][%{{.*}}]
// CHECK:       return %[[VAL]] : f32
func.func @const_stencil_load(%mem: memref<512xf32>,
    %i_diff: index, %j_diff: index, %k_diff: index,
    %radius: index) -> f32 {
  %c8 = arith.constant 8 : index
  %row = lego.row [%c8, %c8, %c8] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[%c8, %c8, %c8]] : !lego.layout
  %view = lego.cast_view %mem, %tiled : memref<512xf32>, !lego.layout -> !lego.view<f32>
  %a = arith.addi %i_diff, %radius : index
  %b = arith.addi %j_diff, %radius : index
  %c = arith.addi %k_diff, %radius : index
  %val = lego.load %view[%a, %b, %c] : !lego.view<f32>, index, index, index
  return %val : f32
}
