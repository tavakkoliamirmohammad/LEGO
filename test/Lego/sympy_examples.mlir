// RUN: lego-opt %s -lego-lower -split-input-file | FileCheck %s
//
// ============================================================================
// Tests derived from Python/SymPy layout examples
//
// SymPy-verified expressions (run via python/lego/lego.py):
//   graphene:   i + 8*j + 2*k + 16*q + 4*w
//   const_out:  64*a + 8*b + c
//   lud_fwd:    32*ii + 16*jj + tidx + 4*tidy
//   normal_out: (bx*2+i)*64 + (by*2+j)*8 + (bz*2+k)
//   bricks_out: (bx*16+by*4+bz)*8 + (i*4+j*2+k)
// ============================================================================

// -----

// const = OrderBy(Row(8, 8, 8)).TileBy([8, 8, 8])
// SymPy: 64*a + 8*b + c

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

// graphene.py: OrderBy(RegP([2,2,2,2,2], [4,1,3,2,0])).GroupBy([(2,2,2,2,2)])
// SymPy: i + 2*k + 4*w + 8*j + 16*q
//
// With assume_bounds + materialize-assume-bounds + iterated A2/int-range,
// the flatten/unflatten chain fully simplifies to the permuted expression.

// CHECK-LABEL: func.func @graphene_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index, %[[W:.*]]: index, %[[Q:.*]]: index)
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[MK:.*]] = arith.muli %[[K]], %[[C2]] : index
// CHECK:       %[[S1:.*]] = arith.addi %[[I]], %[[MK]] : index
// CHECK:       %[[MW:.*]] = arith.muli %[[W]], %[[C4]] : index
// CHECK:       %[[S2:.*]] = arith.addi %[[S1]], %[[MW]] : index
// CHECK:       %[[MJ:.*]] = arith.muli %[[J]], %[[C8]] : index
// CHECK:       %[[S3:.*]] = arith.addi %[[S2]], %[[MJ]] : index
// CHECK:       %[[MQ:.*]] = arith.muli %[[Q]], %[[C16]] : index
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

// normal = OrderBy(Row(8,8,8)).TileBy([4,4,4], [2,2,2])
// SymPy: (bx*2+i)*64 + (by*2+j)*8 + (bz*2+k)
//
// Inner has 1 block, tile has 2 levels → general GroupBy path.
// The lowered form retains divui/remui (modular arithmetic equivalent).

// CHECK-LABEL: func.func @normal_3d_apply
// CHECK-SAME:  (%[[BX:.*]]: index, %[[BY:.*]]: index, %[[BZ:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C128:.*]] = arith.constant 128 : index
// CHECK:       arith.muli
// CHECK:       arith.addi
// CHECK:       arith.divui
// CHECK:       arith.remui
// CHECK:       arith.muli %[[BX]], %[[C128]] : index
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

// bricks = OrderBy(Row(4,4,4), Row(2,2,2)).TileBy([4,4,4],[2,2,2])
// SymPy: (bx*16 + by*4 + bz) * 8 + (i*4 + j*2 + k)
//
// TileBy with tile_dims matching inner_dims is identity.

// CHECK-LABEL: func.func @bricks_3d_apply
// CHECK-SAME:  (%[[BX:.*]]: index, %[[BY:.*]]: index, %[[BZ:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[MBY:.*]] = arith.muli %[[BY]], %[[C4]] : index
// CHECK:       %[[S1:.*]] = arith.addi %[[BZ]], %[[MBY]] : index
// CHECK:       %[[MBX:.*]] = arith.muli %[[BX]], %[[C16]] : index
// CHECK:       %[[BLOCK:.*]] = arith.addi %[[S1]], %[[MBX]] : index
// CHECK:       %[[MJ:.*]] = arith.muli %[[J]], %[[C2]] : index
// CHECK:       %[[S2:.*]] = arith.addi %[[K]], %[[MJ]] : index
// CHECK:       %[[MI:.*]] = arith.muli %[[I]], %[[C4]] : index
// CHECK:       %[[LOCAL:.*]] = arith.addi %[[S2]], %[[MI]] : index
// CHECK:       %[[SCALED:.*]] = arith.muli %[[BLOCK]], %[[C8]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[SCALED]], %[[LOCAL]] : index
// CHECK:       return %[[RES]] : index
func.func @bricks_3d_apply(%bx: index, %by: index, %bz: index,
                           %i: index, %j: index, %k: index) -> index {
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %row1 = lego.row [%c4, %c4, %c4] : !lego.layout
  %row2 = lego.row [%c2, %c2, %c2] : !lego.layout
  %ob = lego.order_by(%row1, %row2) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[%c4, %c4, %c4], [%c2, %c2, %c2]] : !lego.layout
  %f = lego.apply %tiled(%bx, %by, %bz, %i, %j, %k) : !lego.layout
  return %f : index
}

// -----

// lud.py: OrderBy(Row(R*T, R*T)).GroupBy([(R, R), (T, T)])
// SymPy: ii*(R*T*T) + jj*(T*T) + tidy*T + tidx
// With symbolic R and T (matching the Python API).

// CHECK-LABEL: func.func @lud_groupby_apply
// CHECK-SAME:  (%[[R:.*]]: index, %[[T:.*]]: index, %[[II:.*]]: index, %[[JJ:.*]]: index, %[[TIDY:.*]]: index, %[[TIDX:.*]]: index)
// CHECK:       %[[MT:.*]] = arith.muli %[[TIDY]], %[[T]] : index
// CHECK:       %[[S1:.*]] = arith.addi %[[TIDX]], %[[MT]] : index
// CHECK:       %[[TT:.*]] = arith.muli %[[T]], %[[T]] : index
// CHECK:       %[[MJJ:.*]] = arith.muli %[[JJ]], %[[TT]] : index
// CHECK:       %[[S2:.*]] = arith.addi %[[S1]], %[[MJJ]] : index
// CHECK:       %[[TTR:.*]] = arith.muli %[[TT]], %[[R]] : index
// CHECK:       %[[MII:.*]] = arith.muli %[[II]], %[[TTR]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[S2]], %[[MII]] : index
// CHECK:       return %[[RES]] : index
func.func @lud_groupby_apply(%R: index, %T: index,
                             %ii: index, %jj: index,
                             %tidy: index, %tidx: index) -> index {
  %RT = arith.muli %R, %T : index
  %row = lego.row [%RT, %RT] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %gb = lego.group_by [%R, %R, %T, %T](%ob) : !lego.layout
  %f = lego.apply %gb(%ii, %jj, %tidy, %tidx) : !lego.layout
  return %f : index
}

// -----

// lud.py inverse: l.inv((ii*R+jj)*T*T + tid) = [ii, jj, tid/T, tid%T]
//
// Python: expr = (ii * R + jj) * T * T + tid
//         constraints = [ii < R, jj < R, tid < T*T, 0 <= tid]
//         i, j, tidy, tidx = l.inv(expr)
// SymPy result: i=ii, j=jj, tidy=tid/T, tidx=tid%T

// CHECK-LABEL: func.func @lud_groupby_inv
// CHECK-SAME:  (%[[R:.*]]: index, %[[T:.*]]: index, %[[II:.*]]: index, %[[JJ:.*]]: index, %[[TID:.*]]: index)
// CHECK:       %[[RT:.*]] = arith.muli %[[R]], %[[T]] : index
// CHECK:       %[[TT:.*]] = arith.muli %[[T]], %[[T]] : index
// CHECK:       arith.muli %[[II]], %[[R]] : index
// CHECK:       arith.addi {{.*}}, %[[JJ]] : index
// CHECK:       arith.muli {{.*}}, %[[TT]] : index
// CHECK:       arith.addi {{.*}}, %[[TID]] : index
// CHECK:       arith.muli %[[RT]], %[[RT]] : index
// CHECK:       arith.remui
// CHECK:       arith.muli %[[RT]], %[[T]] : index
// CHECK:       arith.divui
// CHECK:       arith.remui
// CHECK:       arith.divui {{.*}}, %[[TT]] : index
// CHECK:       arith.remui {{.*}}, %[[TT]] : index
// CHECK:       arith.divui {{.*}}, %[[T]] : index
// CHECK:       arith.remui {{.*}}, %[[T]] : index
// CHECK:       return
func.func @lud_groupby_inv(%R: index, %T: index,
                           %ii: index, %jj: index, %tid: index)
    -> (index, index, index, index) {
  %c0 = arith.constant 0 : index
  %RT = arith.muli %R, %T : index
  %TT = arith.muli %T, %T : index
  // assume_bounds: ii < R, jj < R, 0 <= tid < T*T
  lego.assume_bounds %ii lb: %c0 ub: %R
  lego.assume_bounds %jj lb: %c0 ub: %R
  lego.assume_bounds %tid lb: %c0 ub: %TT
  // expr = (ii*R + jj)*T*T + tid
  %iiR = arith.muli %ii, %R : index
  %iiRjj = arith.addi %iiR, %jj : index
  %iiRjjTT = arith.muli %iiRjj, %TT : index
  %expr = arith.addi %iiRjjTT, %tid : index
  // l.inv(expr)
  %row = lego.row [%RT, %RT] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %gb = lego.group_by [%R, %R, %T, %T](%ob) : !lego.layout
  %i, %j, %tidy, %tidx = lego.apply_inverse %gb(%expr) : !lego.layout -> index, index, index, index
  return %i, %j, %tidy, %tidx : index, index, index, index
}

// -----

// Stencil loads

// CHECK-LABEL: func.func @normal_stencil_load
// CHECK:       arith.addi
// CHECK:       arith.muli
// CHECK:       %[[VAL:.*]] = memref.load
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
// CHECK:       arith.addi
// CHECK:       arith.muli
// CHECK:       %[[VAL:.*]] = memref.load
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
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       arith.addi
// CHECK:       arith.muli
// CHECK:       %[[VAL:.*]] = memref.load
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
