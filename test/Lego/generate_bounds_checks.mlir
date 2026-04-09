// RUN: lego-opt %s -lego-generate-bounds-checks | FileCheck %s

// ============================================================================
// Tests for lego-generate-bounds-checks pass in isolation.
//
// This pass inserts lego.assert_apply_bounds before each lego.apply
// and lego.assert_inv_bounds before each lego.apply_inverse.
// ============================================================================

// --- apply: assert_apply_bounds is inserted before apply ---
// CHECK-LABEL: func.func @bounds_check_apply
// CHECK-SAME: (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK:       %[[LAYOUT:.*]] = lego.row
// CHECK:       lego.assert_apply_bounds %[[LAYOUT]](%[[I]], %[[J]])
// CHECK-NEXT:  lego.apply %[[LAYOUT]](%[[I]], %[[J]])
func.func @bounds_check_apply(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %layout = lego.row [%c4, %c8] : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// --- apply_inverse: assert_inv_bounds is inserted before apply_inverse ---
// CHECK-LABEL: func.func @bounds_check_apply_inverse
// CHECK-SAME: (%[[F:.*]]: index)
// CHECK:       %[[LAYOUT:.*]] = lego.row
// CHECK:       lego.assert_inv_bounds %[[LAYOUT]](%[[F]])
// CHECK-NEXT:  lego.apply_inverse %[[LAYOUT]](%[[F]])
func.func @bounds_check_apply_inverse(%f: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %layout = lego.row [%c4, %c8] : !lego.layout
  %i, %j = lego.apply_inverse %layout(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// --- Both apply and apply_inverse in same function ---
// CHECK-LABEL: func.func @bounds_check_both
// CHECK:       lego.assert_apply_bounds
// CHECK:       lego.apply {{.*}}(
// CHECK:       lego.assert_inv_bounds
// CHECK:       lego.apply_inverse
func.func @bounds_check_both(%i: index, %j: index, %f: index) -> (index, index, index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %layout = lego.row [%c4, %c8] : !lego.layout
  %flat = lego.apply %layout(%i, %j) : !lego.layout
  %ri, %rj = lego.apply_inverse %layout(%f) : !lego.layout -> index, index
  return %flat, %ri, %rj : index, index, index
}

// --- 3D layout: assert_apply_bounds with 3 indices ---
// CHECK-LABEL: func.func @bounds_check_3d
// CHECK:       lego.assert_apply_bounds %{{.*}}(%{{.*}}, %{{.*}}, %{{.*}})
// CHECK:       lego.apply %{{.*}}(%{{.*}}, %{{.*}}, %{{.*}})
func.func @bounds_check_3d(%i: index, %j: index, %k: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %layout = lego.row [%c2, %c3, %c4] : !lego.layout
  %f = lego.apply %layout(%i, %j, %k) : !lego.layout
  return %f : index
}

// --- RegP layout: bounds checks should also be inserted ---
// CHECK-LABEL: func.func @bounds_check_regp
// CHECK:       lego.assert_apply_bounds
// CHECK:       lego.apply
func.func @bounds_check_regp(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %layout = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// --- Col layout: bounds checks should be inserted ---
// CHECK-LABEL: func.func @bounds_check_col
// CHECK:       lego.assert_apply_bounds
// CHECK:       lego.apply
func.func @bounds_check_col(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %layout = lego.col [%c4, %c8] : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// --- Multiple apply ops: each gets its own bounds check ---
// CHECK-LABEL: func.func @bounds_check_multiple_apply
// CHECK:       lego.assert_apply_bounds
// CHECK:       lego.apply
// CHECK:       lego.assert_apply_bounds
// CHECK:       lego.apply
func.func @bounds_check_multiple_apply(%i: index, %j: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %r = lego.row [%c4, %c8] : !lego.layout
  %c = lego.col [%c4, %c8] : !lego.layout
  %f1 = lego.apply %r(%i, %j) : !lego.layout
  %f2 = lego.apply %c(%i, %j) : !lego.layout
  return %f1, %f2 : index, index
}

// --- GenP layout: bounds checks should be inserted ---
// CHECK-LABEL: func.func @bounds_check_genp
// CHECK:       lego.assert_apply_bounds
// CHECK:       lego.apply
func.func @bounds_check_genp(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %layout = lego.gen_p [%c4, %c4] apply (%a: index, %b: index) {
    %sum = arith.addi %a, %b : index
    lego.yield %sum : index
  } inv (%flat: index) {
    %z = arith.constant 0 : index
    lego.yield %z, %z : index, index
  } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}
