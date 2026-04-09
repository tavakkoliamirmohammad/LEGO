// RUN: lego-opt %s -lego-lower -split-input-file | FileCheck %s

// ============================================================================
// Edge-case tests for the full lego-lower pipeline.
//
// Covers: 1D layouts, size-1 dimensions, deeply nested layouts,
// and mixed symbolic/constant dimensions through the full pipeline.
// ============================================================================

// -----

// --- 1D Row apply: identity mapping ---
// CHECK-LABEL: func.func @row_1d_lower
// CHECK-SAME:  (%[[I:.*]]: index)
// No arithmetic needed for 1D identity.
// CHECK:       return %[[I]] : index
func.func @row_1d_lower(%i: index) -> index {
  %c16 = arith.constant 16 : index
  %r = lego.row [%c16] : !lego.layout
  %f = lego.apply %r(%i) : !lego.layout
  return %f : index
}

// -----

// --- 1D Col apply: also identity ---
// CHECK-LABEL: func.func @col_1d_lower
// CHECK-SAME:  (%[[I:.*]]: index)
// CHECK:       return %[[I]] : index
func.func @col_1d_lower(%i: index) -> index {
  %c16 = arith.constant 16 : index
  %c = lego.col [%c16] : !lego.layout
  %f = lego.apply %c(%i) : !lego.layout
  return %f : index
}

// -----

// --- 1D Row inverse: identity ---
// CHECK-LABEL: func.func @row_1d_inv_lower
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK:       return %[[F]] : index
func.func @row_1d_inv_lower(%f: index) -> index {
  %c16 = arith.constant 16 : index
  %r = lego.row [%c16] : !lego.layout
  %i = lego.apply_inverse %r(%f) : !lego.layout -> index
  return %i : index
}

// -----

// --- Size-1 dimension in 2D Row: Row(1, N) -> flat = j ---
// CHECK-LABEL: func.func @row_size1_first_dim
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// The muli by N and addi with j should simplify, but at minimum no div/rem.
// CHECK-NOT:   arith.divui
// CHECK-NOT:   arith.remui
// CHECK:       return
func.func @row_size1_first_dim(%i: index, %j: index) -> index {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %r = lego.row [%c1, %c8] : !lego.layout
  %f = lego.apply %r(%i, %j) : !lego.layout
  return %f : index
}

// -----

// --- OrderBy with single 1D block: trivial ---
// CHECK-LABEL: func.func @orderby_single_1d
// CHECK-SAME:  (%[[I:.*]]: index)
// CHECK:       return %[[I]] : index
func.func @orderby_single_1d(%i: index) -> index {
  %c10 = arith.constant 10 : index
  %p = lego.reg_p perm [0] dims [%c10] : !lego.layout
  %ob = lego.order_by(%p) : !lego.layout
  %f = lego.apply %ob(%i) : !lego.layout
  return %f : index
}

// -----

// --- TileBy identity: tile dims match inner block dims ---
// TileBy(OrderBy(Row(6, 8)), [[6, 8]]) -> identity (no div/rem)
// CHECK-LABEL: func.func @tileby_identity_lower
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// Should lower to simple row-major: i * 8 + j (with strength reduction)
// CHECK-NOT:   arith.divui
// CHECK-NOT:   arith.remui
// CHECK:       return
func.func @tileby_identity_lower(%i: index, %j: index) -> index {
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %r = lego.row [%c6, %c8] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout
  %tb = lego.tile_by %ob tile_dims [[%c6, %c8]] : !lego.layout
  %f = lego.apply %tb(%i, %j) : !lego.layout
  return %f : index
}

// -----

// --- GroupBy that re-groups a transposed layout ---
// GroupBy(OrderBy(RegP([4,8], [1,0])), [8,4]) -> col-major regrouped
// CHECK-LABEL: func.func @groupby_regroup_lower
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// Should produce arithmetic, not lego ops.
// CHECK-NOT:   lego.
// CHECK:       return
func.func @groupby_regroup_lower(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rp = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  %ob = lego.order_by(%rp) : !lego.layout
  %gb = lego.group_by [%c8, %c4](%ob) : !lego.layout
  %f = lego.apply %gb(%i, %j) : !lego.layout
  return %f : index
}

// -----

// --- Full roundtrip through pipeline: apply then apply_inverse should cancel ---
// CHECK-LABEL: func.func @roundtrip_cancel
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// After CSE/canonicalize, apply followed by apply_inverse should produce
// the original indices (or equivalent arithmetic that simplifies).
// CHECK-NOT:   lego.
// CHECK:       return
func.func @roundtrip_cancel(%i: index, %j: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %r = lego.row [%c4, %c8] : !lego.layout
  %flat = lego.apply %r(%i, %j) : !lego.layout
  %ri, %rj = lego.apply_inverse %r(%flat) : !lego.layout -> index, index
  return %ri, %rj : index, index
}

// -----

// --- assume_bounds are consumed by the pipeline ---
// CHECK-LABEL: func.func @assume_consumed
// CHECK-NOT:   lego.assume_bounds
// CHECK-NOT:   lego.assume
// CHECK:       return
func.func @assume_consumed(%x: index) -> index {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  lego.assume_bounds %x lb: %c0 ub: %c10
  %r = lego.row [%c10] : !lego.layout
  %f = lego.apply %r(%x) : !lego.layout
  return %f : index
}

// -----

// --- Power-of-2 layout gets strength-reduced through the pipeline ---
// Row(8, 8): apply(i, j) = i*8 + j -> (i << 3) + j
// CHECK-LABEL: func.func @pow2_strength_reduced
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK:       arith.shli %[[I]], %{{.*}} : index
// CHECK:       arith.addi
// CHECK:       return
func.func @pow2_strength_reduced(%i: index, %j: index) -> index {
  %c8 = arith.constant 8 : index
  %r = lego.row [%c8, %c8] : !lego.layout
  %f = lego.apply %r(%i, %j) : !lego.layout
  return %f : index
}

// -----

// --- Power-of-2 inverse: Row(8, 8) inv -> shrui and andi ---
// CHECK-LABEL: func.func @pow2_inv_strength_reduced
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK:       arith.shrui %[[F]], %{{.*}} : index
// CHECK:       arith.andi %[[F]], %{{.*}} : index
// CHECK:       return
func.func @pow2_inv_strength_reduced(%f: index) -> (index, index) {
  %c8 = arith.constant 8 : index
  %r = lego.row [%c8, %c8] : !lego.layout
  %i, %j = lego.apply_inverse %r(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}
