// RUN: lego-opt %s -lego-normalization | FileCheck %s

// ============================================================================
// Edge-case tests for the lego-normalization pass.
//
// Covers: 1D Row/Col, 3D Col, 4D Row, assume_bounds variants,
// and TileBy with non-square dimensions.
// ============================================================================

// --- 1D Row normalizes to RegP with identity perm [0] ---
// CHECK-LABEL: func @norm_row_1d
// CHECK:       %[[REGP:.*]] = lego.reg_p perm [0] dims[%{{.*}}]
// CHECK:       return %[[REGP]]
func.func @norm_row_1d() -> !lego.layout {
  %c16 = arith.constant 16 : index
  %r = lego.row [%c16] : !lego.layout
  return %r : !lego.layout
}

// --- 1D Col normalizes to RegP with perm [0] (reversed identity of size 1 is still [0]) ---
// CHECK-LABEL: func @norm_col_1d
// CHECK:       %[[REGP:.*]] = lego.reg_p perm [0] dims[%{{.*}}]
// CHECK:       return %[[REGP]]
func.func @norm_col_1d() -> !lego.layout {
  %c16 = arith.constant 16 : index
  %c = lego.col [%c16] : !lego.layout
  return %c : !lego.layout
}

// --- 3D Col normalizes to RegP with reversed perm [2, 1, 0] ---
// CHECK-LABEL: func @norm_col_3d
// CHECK:       %[[REGP:.*]] = lego.reg_p perm [2, 1, 0] dims[%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:       return %[[REGP]]
func.func @norm_col_3d() -> !lego.layout {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c = lego.col [%c2, %c3, %c4] : !lego.layout
  return %c : !lego.layout
}

// --- 4D Row normalizes to RegP with identity perm [0, 1, 2, 3] ---
// CHECK-LABEL: func @norm_row_4d
// CHECK:       %[[REGP:.*]] = lego.reg_p perm [0, 1, 2, 3] dims[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:       return %[[REGP]]
func.func @norm_row_4d() -> !lego.layout {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %r = lego.row [%c2, %c3, %c4, %c5] : !lego.layout
  return %r : !lego.layout
}

// --- 4D Col normalizes to RegP with reversed perm [3, 2, 1, 0] ---
// CHECK-LABEL: func @norm_col_4d
// CHECK:       %[[REGP:.*]] = lego.reg_p perm [3, 2, 1, 0] dims[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:       return %[[REGP]]
func.func @norm_col_4d() -> !lego.layout {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c = lego.col [%c2, %c3, %c4, %c5] : !lego.layout
  return %c : !lego.layout
}

// --- assume_bounds with both lb and ub normalizes to two assume ops ---
// CHECK-LABEL: func.func @norm_assume_both
// CHECK-SAME:  (%[[VAL:arg[0-9]+]]: index, %[[LB:arg[0-9]+]]: index, %[[UB:arg[0-9]+]]: index)
// CHECK:       %[[CMP_GE:.*]] = arith.cmpi sge, %[[VAL]], %[[LB]] : index
// CHECK:       lego.assume %[[CMP_GE]]
// CHECK:       %[[CMP_LT:.*]] = arith.cmpi slt, %[[VAL]], %[[UB]] : index
// CHECK:       lego.assume %[[CMP_LT]]
// CHECK:       return
func.func @norm_assume_both(%val: index, %lb: index, %ub: index) {
  lego.assume_bounds %val lb: %lb ub: %ub
  return
}

// --- Multiple assume_bounds on same value ---
// CHECK-LABEL: func.func @norm_multiple_assumes
// CHECK-SAME:  (%[[VAL:arg[0-9]+]]: index, %[[LB:arg[0-9]+]]: index, %[[UB1:arg[0-9]+]]: index, %[[UB2:arg[0-9]+]]: index)
// CHECK:       arith.cmpi sge, %[[VAL]], %[[LB]]
// CHECK:       lego.assume
// CHECK:       arith.cmpi slt, %[[VAL]], %[[UB1]]
// CHECK:       lego.assume
// CHECK:       arith.cmpi slt, %[[VAL]], %[[UB2]]
// CHECK:       lego.assume
// CHECK:       return
func.func @norm_multiple_assumes(%val: index, %lb: index, %ub1: index, %ub2: index) {
  lego.assume_bounds %val lb: %lb ub: %ub1
  lego.assume_bounds %val ub: %ub2
  return
}

// --- Row with symbolic dimension normalizes to RegP ---
// CHECK-LABEL: func @norm_row_symbolic
// CHECK-SAME:  (%[[D:.*]]: index)
// CHECK:       %[[REGP:.*]] = lego.reg_p perm [0] dims[%[[D]]]
// CHECK:       return %[[REGP]]
func.func @norm_row_symbolic(%d: index) -> !lego.layout {
  %r = lego.row [%d] : !lego.layout
  return %r : !lego.layout
}

// --- Col with symbolic dimensions normalizes to RegP ---
// CHECK-LABEL: func @norm_col_symbolic
// CHECK-SAME:  (%[[D0:.*]]: index, %[[D1:.*]]: index)
// CHECK:       %[[REGP:.*]] = lego.reg_p perm [1, 0] dims[%[[D0]], %[[D1]]]
// CHECK:       return %[[REGP]]
func.func @norm_col_symbolic(%d0: index, %d1: index) -> !lego.layout {
  %c = lego.col [%d0, %d1] : !lego.layout
  return %c : !lego.layout
}

// --- TileBy with non-square dimensions: different tile sizes per dim ---
// TileBy(OrderBy(Row(12, 8)), [[3, 2], [4, 4]])
// d=2, q=2 -> sigma=[0, 2, 1, 3]
// CHECK-LABEL: func @norm_tileby_nonsquare
// CHECK:       lego.reg_p perm [0, 1] dims[%{{.*}}, %{{.*}}]
// CHECK:       lego.order_by
// The tile reshuffle permutation: d=2, q=2 -> [0, 2, 1, 3]
// CHECK:       lego.reg_p perm [0, 2, 1, 3] dims[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:       lego.group_by
func.func @norm_tileby_nonsquare() -> !lego.layout {
  %c12 = arith.constant 12 : index
  %c8 = arith.constant 8 : index
  %r = lego.row [%c12, %c8] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %tb = lego.tile_by %ob tile_dims [[%c3, %c2], [%c4, %c4]] : !lego.layout
  return %tb : !lego.layout
}

// --- RegP ops should NOT be rewritten by normalization (pass-through) ---
// CHECK-LABEL: func @norm_regp_passthrough
// CHECK:       %[[REGP:.*]] = lego.reg_p perm [1, 0] dims[%{{.*}}, %{{.*}}]
// CHECK:       return %[[REGP]]
func.func @norm_regp_passthrough() -> !lego.layout {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rp = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  return %rp : !lego.layout
}

// --- OrderBy ops should NOT be rewritten by normalization (inner ops get normalized) ---
// CHECK-LABEL: func @norm_orderby_inner
// The Row inside OrderBy is normalized to RegP, but OrderBy stays.
// CHECK:       %[[RP1:.*]] = lego.reg_p perm [0] dims[%{{.*}}]
// CHECK:       %[[RP2:.*]] = lego.reg_p perm [0] dims[%{{.*}}]
// CHECK:       %[[OB:.*]] = lego.order_by(%[[RP1]], %[[RP2]])
// CHECK:       return %[[OB]]
func.func @norm_orderby_inner() -> !lego.layout {
  %c4 = arith.constant 4 : index
  %r1 = lego.row [%c4] : !lego.layout
  %c8 = arith.constant 8 : index
  %r2 = lego.row [%c8] : !lego.layout
  %ob = lego.order_by(%r1, %r2) : !lego.layout
  return %ob : !lego.layout
}

// --- GroupBy wrapping Row ops: inner Row is normalized ---
// CHECK-LABEL: func @norm_groupby_inner
// CHECK:       lego.reg_p perm [0, 1] dims
// CHECK:       lego.order_by
// CHECK:       lego.group_by
func.func @norm_groupby_inner() -> !lego.layout {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %r = lego.row [%c4, %c8] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout
  %gb = lego.group_by [%c4, %c8](%ob) : !lego.layout
  return %gb : !lego.layout
}
