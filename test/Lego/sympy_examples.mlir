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
//
// After strength reduction, power-of-2 muli become shli.
// ============================================================================

// -----

// const = OrderBy(Row(8, 8, 8)).TileBy([8, 8, 8])
// SymPy: 64*a + 8*b + c  →  (a << 6) + (b << 3) + c

// CHECK-LABEL: func.func @const_3d_apply
// CHECK-SAME:  (%[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index)
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK:       %[[T1:.*]] = arith.shli %[[B]], %[[C3]] : index
// CHECK:       %[[T2:.*]] = arith.addi %[[C]], %[[T1]] : index
// CHECK:       %[[T3:.*]] = arith.shli %[[A]], %[[C6]] : index
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
// SymPy: i + 2*k + 4*w + 8*j + 16*q  →  i + (k<<1) + (w<<2) + (j<<3) + (q<<4)

// CHECK-LABEL: func.func @graphene_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index, %[[W:.*]]: index, %[[Q:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[MK:.*]] = arith.shli %[[K]], %[[C1]] : index
// CHECK:       %[[S1:.*]] = arith.addi %[[I]], %[[MK]] : index
// CHECK:       %[[MW:.*]] = arith.shli %[[W]], %[[C2]] : index
// CHECK:       %[[S2:.*]] = arith.addi %[[S1]], %[[MW]] : index
// CHECK:       %[[MJ:.*]] = arith.shli %[[J]], %[[C3]] : index
// CHECK:       %[[S3:.*]] = arith.addi %[[S2]], %[[MJ]] : index
// CHECK:       %[[MQ:.*]] = arith.shli %[[Q]], %[[C4]] : index
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
//      → (bx*2+i)<<6 + (by*2+j)<<3 + (bz*2+k)

// CHECK-LABEL: func.func @normal_3d_apply
// CHECK-SAME:  (%[[BX:.*]]: index, %[[BY:.*]]: index, %[[BZ:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[MBX:.*]] = arith.shli %[[BX]], %[[C1]] : index
// CHECK:       %[[CX:.*]] = arith.addi %[[I]], %[[MBX]] : index
// CHECK:       %[[MBY:.*]] = arith.shli %[[BY]], %[[C1]] : index
// CHECK:       %[[CY:.*]] = arith.addi %[[J]], %[[MBY]] : index
// CHECK:       %[[MBZ:.*]] = arith.shli %[[BZ]], %[[C1]] : index
// CHECK:       %[[CZ:.*]] = arith.addi %[[K]], %[[MBZ]] : index
// CHECK:       %[[S1:.*]] = arith.shli %[[CY]], %[[C3]] : index
// CHECK:       %[[S2:.*]] = arith.addi %[[CZ]], %[[S1]] : index
// CHECK:       %[[S3:.*]] = arith.shli %[[CX]], %[[C6]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[S2]], %[[S3]] : index
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

// Identity TileBy → passes through to inner OrderBy
// Result: (bx<<4 + by<<2 + bz) << 3 + (i<<2 + j<<1 + k)
// CHECK-LABEL: func.func @bricks_3d_apply
// CHECK-SAME:  (%[[BX:.*]]: index, %[[BY:.*]]: index, %[[BZ:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[MBY:.*]] = arith.shli %[[BY]], %[[C2]] : index
// CHECK:       %[[S1:.*]] = arith.addi %[[BZ]], %[[MBY]] : index
// CHECK:       %[[MBX:.*]] = arith.shli %[[BX]], %[[C4]] : index
// CHECK:       %[[BLOCK:.*]] = arith.addi %[[S1]], %[[MBX]] : index
// CHECK:       %[[MJ:.*]] = arith.shli %[[J]], %[[C1]] : index
// CHECK:       %[[S2:.*]] = arith.addi %[[K]], %[[MJ]] : index
// CHECK:       %[[MI:.*]] = arith.shli %[[I]], %[[C2]] : index
// CHECK:       %[[LOCAL:.*]] = arith.addi %[[S2]], %[[MI]] : index
// CHECK:       %[[SCALED:.*]] = arith.shli %[[BLOCK]], %[[C3]] : index
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
// With symbolic R and T — no strength reduction (not constant power-of-2).

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
  %c1 = arith.constant 1 : index
  lego.assume_bounds %R lb: %c1
  lego.assume_bounds %T lb: %c1
  %RT = arith.muli %R, %T : index
  %row = lego.row [%RT, %RT] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %gb = lego.group_by [%R, %R, %T, %T](%ob) : !lego.layout
  %f = lego.apply %gb(%ii, %jj, %tidy, %tidx) : !lego.layout
  return %f : index
}

// -----

// lud.py inverse: l.inv((ii*R+jj)*T*T + tid) = [ii, jj, tid/T, tid%T]
// Symbolic divisor T — no strength reduction.

// CHECK-LABEL: func.func @lud_groupby_inv
// CHECK-SAME:  (%[[R:.*]]: index, %[[T:.*]]: index, %[[II:.*]]: index, %[[JJ:.*]]: index, %[[TID:.*]]: index)
// CHECK:       %[[OUT2:.*]] = arith.divui %[[TID]], %[[T]] : index
// CHECK:       %[[OUT3:.*]] = arith.remui %[[TID]], %[[T]] : index
// CHECK:       return %[[II]], %[[JJ]], %[[OUT2]], %[[OUT3]] : index, index, index, index
func.func @lud_groupby_inv(%R: index, %T: index,
                           %ii: index, %jj: index, %tid: index)
    -> (index, index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %RT = arith.muli %R, %T : index
  %TT = arith.muli %T, %T : index
  // assume_bounds: R > 0, T > 0, ii < R, jj < R, 0 <= tid < T*T
  lego.assume_bounds %R lb: %c1
  lego.assume_bounds %T lb: %c1
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

// Direct TileBy lowering: (bx*2+io)<<6 + (by*2+jo)<<3 + (bz*2+ko)
// CHECK-LABEL: func.func @normal_stencil_load
// CHECK-SAME:  (%[[MEM:.*]]: memref<512xf32>, %[[BX:.*]]: index, %[[BY:.*]]: index, %[[BZ:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index, %[[ID:.*]]: index, %[[JD:.*]]: index, %[[KD:.*]]: index)
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[IO:.*]] = arith.addi %[[I]], %[[ID]] : index
// CHECK:       %[[JO:.*]] = arith.addi %[[J]], %[[JD]] : index
// CHECK:       %[[KO:.*]] = arith.addi %[[K]], %[[KD]] : index
// CHECK:       %[[MBX:.*]] = arith.shli %[[BX]], %[[C1]] : index
// CHECK:       %[[CX:.*]] = arith.addi %[[IO]], %[[MBX]] : index
// CHECK:       %[[MBY:.*]] = arith.shli %[[BY]], %[[C1]] : index
// CHECK:       %[[CY:.*]] = arith.addi %[[JO]], %[[MBY]] : index
// CHECK:       %[[MBZ:.*]] = arith.shli %[[BZ]], %[[C1]] : index
// CHECK:       %[[CZ:.*]] = arith.addi %[[KO]], %[[MBZ]] : index
// CHECK:       %[[S1:.*]] = arith.shli %[[CY]], %[[C3]] : index
// CHECK:       %[[S2:.*]] = arith.addi %[[CZ]], %[[S1]] : index
// CHECK:       %[[S3:.*]] = arith.shli %[[CX]], %[[C6]] : index
// CHECK:       %[[IDX:.*]] = arith.addi %[[S2]], %[[S3]] : index
// CHECK:       %[[VAL:.*]] = memref.load %[[MEM]][%[[IDX]]] : memref<512xf32>
// CHECK:       return %[[VAL]] : f32
func.func @normal_stencil_load(%mem: memref<512xf32>,
    %bx: index, %by: index, %bz: index,
    %i: index, %j: index, %k: index,
    %i_diff: index, %j_diff: index, %k_diff: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  lego.assume_bounds %bx lb: %c0 ub: %c4
  lego.assume_bounds %by lb: %c0 ub: %c4
  lego.assume_bounds %bz lb: %c0 ub: %c4
  %row = lego.row [%c8, %c8, %c8] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[%c4, %c4, %c4], [%c2, %c2, %c2]] : !lego.layout
  %view = lego.cast_view %mem, %tiled : memref<512xf32>, !lego.layout -> !lego.view<f32>
  %io = arith.addi %i, %i_diff : index
  %jo = arith.addi %j, %j_diff : index
  %ko = arith.addi %k, %k_diff : index
  lego.assume_bounds %io lb: %c0 ub: %c2
  lego.assume_bounds %jo lb: %c0 ub: %c2
  lego.assume_bounds %ko lb: %c0 ub: %c2
  %val = lego.load %view[%bx, %by, %bz, %io, %jo, %ko] : !lego.view<f32>, index, index, index, index, index, index
  return %val : f32
}

// -----

// CHECK-LABEL: func.func @bricks_stencil_load
// CHECK-SAME:  (%[[MEM:.*]]: memref<512xf32>, %[[BX:.*]]: index, %[[BY:.*]]: index, %[[BZ:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index, %[[ID:.*]]: index, %[[JD:.*]]: index, %[[KD:.*]]: index)
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[IO:.*]] = arith.addi %[[I]], %[[ID]] : index
// CHECK:       %[[JO:.*]] = arith.addi %[[J]], %[[JD]] : index
// CHECK:       %[[KO:.*]] = arith.addi %[[K]], %[[KD]] : index
// CHECK:       %[[MBY:.*]] = arith.shli %[[BY]], %[[C2]] : index
// CHECK:       %[[S0:.*]] = arith.addi %[[BZ]], %[[MBY]] : index
// CHECK:       %[[MBX:.*]] = arith.shli %[[BX]], %[[C4]] : index
// CHECK:       %[[BLOCK:.*]] = arith.addi %[[S0]], %[[MBX]] : index
// CHECK:       %[[MJ:.*]] = arith.shli %[[JO]], %[[C1]] : index
// CHECK:       %[[S1:.*]] = arith.addi %[[KO]], %[[MJ]] : index
// CHECK:       %[[MI:.*]] = arith.shli %[[IO]], %[[C2]] : index
// CHECK:       %[[LOCAL:.*]] = arith.addi %[[S1]], %[[MI]] : index
// CHECK:       %[[SCALED:.*]] = arith.shli %[[BLOCK]], %[[C3]] : index
// CHECK:       %[[IDX:.*]] = arith.addi %[[SCALED]], %[[LOCAL]] : index
// CHECK:       %[[VAL:.*]] = memref.load %[[MEM]][%[[IDX]]] : memref<512xf32>
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
// CHECK-SAME:  (%[[MEM:.*]]: memref<512xf32>, %[[IDIFF:.*]]: index, %[[JDIFF:.*]]: index, %[[KDIFF:.*]]: index, %[[RAD:.*]]: index)
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK:       %[[A:.*]] = arith.addi %[[IDIFF]], %[[RAD]] : index
// CHECK:       %[[B:.*]] = arith.addi %[[JDIFF]], %[[RAD]] : index
// CHECK:       %[[C:.*]] = arith.addi %[[KDIFF]], %[[RAD]] : index
// CHECK:       %[[MB:.*]] = arith.shli %[[B]], %[[C3]] : index
// CHECK:       %[[S0:.*]] = arith.addi %[[C]], %[[MB]] : index
// CHECK:       %[[MA:.*]] = arith.shli %[[A]], %[[C6]] : index
// CHECK:       %[[IDX:.*]] = arith.addi %[[S0]], %[[MA]] : index
// CHECK:       %[[VAL:.*]] = memref.load %[[MEM]][%[[IDX]]] : memref<512xf32>
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
