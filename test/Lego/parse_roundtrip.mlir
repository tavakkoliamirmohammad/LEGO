// RUN: lego-opt %s | FileCheck %s
//
// Simple round-trip tests: verify every LEGO op can be parsed and printed
// correctly WITHOUT any lowering pass. This tests ODS definitions,
// custom assembly formats, and type inference.

// ============================================================================
// RowOp
// ============================================================================

// CHECK-LABEL: func.func @parse_row_2d
// CHECK:       lego.row[%{{.*}}, %{{.*}}] : !lego.layout
// CHECK:       lego.apply
func.func @parse_row_2d(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %r = lego.row [%c4, %c8] : !lego.layout
  %f = lego.apply %r(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_row_3d
// CHECK:       lego.row[%{{.*}}, %{{.*}}, %{{.*}}] : !lego.layout
func.func @parse_row_3d(%i: index, %j: index, %k: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %r = lego.row [%c2, %c3, %c4] : !lego.layout
  %f = lego.apply %r(%i, %j, %k) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_row_4d
// CHECK:       lego.row[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : !lego.layout
func.func @parse_row_4d(%i: index, %j: index, %k: index, %l: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %r = lego.row [%c2, %c3, %c4, %c5] : !lego.layout
  %f = lego.apply %r(%i, %j, %k, %l) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_row_1d
// CHECK:       lego.row[%{{.*}}] : !lego.layout
func.func @parse_row_1d(%i: index) -> index {
  %c16 = arith.constant 16 : index
  %r = lego.row [%c16] : !lego.layout
  %f = lego.apply %r(%i) : !lego.layout
  return %f : index
}

// ============================================================================
// ColOp
// ============================================================================

// CHECK-LABEL: func.func @parse_col_2d
// CHECK:       lego.col[%{{.*}}, %{{.*}}] : !lego.layout
// CHECK:       lego.apply
func.func @parse_col_2d(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c = lego.col [%c4, %c8] : !lego.layout
  %f = lego.apply %c(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_col_3d
// CHECK:       lego.col[%{{.*}}, %{{.*}}, %{{.*}}] : !lego.layout
func.func @parse_col_3d(%i: index, %j: index, %k: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c = lego.col [%c2, %c3, %c4] : !lego.layout
  %f = lego.apply %c(%i, %j, %k) : !lego.layout
  return %f : index
}

// ============================================================================
// RegPOp
// ============================================================================

// CHECK-LABEL: func.func @parse_regp_identity
// CHECK:       lego.reg_p perm [0, 1] dims[%{{.*}}, %{{.*}}] : !lego.layout
func.func @parse_regp_identity(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rp = lego.reg_p perm [0, 1] dims [%c4, %c8] : !lego.layout
  %f = lego.apply %rp(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_regp_transpose
// CHECK:       lego.reg_p perm [1, 0] dims[%{{.*}}, %{{.*}}] : !lego.layout
func.func @parse_regp_transpose(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rp = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  %f = lego.apply %rp(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_regp_3d
// CHECK:       lego.reg_p perm [2, 0, 1] dims[%{{.*}}, %{{.*}}, %{{.*}}] : !lego.layout
func.func @parse_regp_3d(%i: index, %j: index, %k: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %rp = lego.reg_p perm [2, 0, 1] dims [%c2, %c3, %c4] : !lego.layout
  %f = lego.apply %rp(%i, %j, %k) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_regp_1d
// CHECK:       lego.reg_p perm [0] dims[%{{.*}}] : !lego.layout
func.func @parse_regp_1d(%i: index) -> index {
  %c10 = arith.constant 10 : index
  %rp = lego.reg_p perm [0] dims [%c10] : !lego.layout
  %f = lego.apply %rp(%i) : !lego.layout
  return %f : index
}

// ============================================================================
// OrderByOp
// ============================================================================

// CHECK-LABEL: func.func @parse_orderby_two
// CHECK:       lego.order_by(%{{.*}}, %{{.*}}) : !lego.layout
func.func @parse_orderby_two(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %p1 = lego.reg_p perm [0] dims [%c4] : !lego.layout
  %c8 = arith.constant 8 : index
  %p2 = lego.reg_p perm [0] dims [%c8] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %f = lego.apply %ob(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_orderby_three
// CHECK:       lego.order_by(%{{.*}}, %{{.*}}, %{{.*}}) : !lego.layout
func.func @parse_orderby_three(%i: index, %j: index, %k: index) -> index {
  %c2 = arith.constant 2 : index
  %p1 = lego.reg_p perm [0] dims [%c2] : !lego.layout
  %c3 = arith.constant 3 : index
  %p2 = lego.reg_p perm [0] dims [%c3] : !lego.layout
  %c4 = arith.constant 4 : index
  %p3 = lego.reg_p perm [0] dims [%c4] : !lego.layout
  %ob = lego.order_by(%p1, %p2, %p3) : !lego.layout
  %f = lego.apply %ob(%i, %j, %k) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_orderby_single
// CHECK:       lego.order_by(%{{.*}}) : !lego.layout
func.func @parse_orderby_single(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rp = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  %ob = lego.order_by(%rp) : !lego.layout
  %f = lego.apply %ob(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_orderby_mixed
// CHECK:       lego.order_by(%{{.*}}, %{{.*}}) : !lego.layout
func.func @parse_orderby_mixed(%i: index, %j: index, %k: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %p1 = lego.reg_p perm [1, 0] dims [%c2, %c3] : !lego.layout
  %c4 = arith.constant 4 : index
  %p2 = lego.reg_p perm [0] dims [%c4] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %f = lego.apply %ob(%i, %j, %k) : !lego.layout
  return %f : index
}

// ============================================================================
// GroupByOp
// ============================================================================

// CHECK-LABEL: func.func @parse_groupby_single
// CHECK:       lego.group_by[%{{.*}}, %{{.*}}] (%{{.*}}) : !lego.layout
func.func @parse_groupby_single(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rp = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  %ob = lego.order_by(%rp) : !lego.layout
  %gb = lego.group_by [%c4, %c8](%ob) : !lego.layout
  %f = lego.apply %gb(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_groupby_multi
// CHECK:       lego.group_by[%{{.*}}] (%{{.*}}, %{{.*}}) : !lego.layout
func.func @parse_groupby_multi(%i: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %rp1 = lego.reg_p perm [1, 0] dims [%c2, %c3] : !lego.layout
  %ob1 = lego.order_by(%rp1) : !lego.layout
  %rp2 = lego.reg_p perm [0, 1] dims [%c2, %c3] : !lego.layout
  %ob2 = lego.order_by(%rp2) : !lego.layout
  %c6 = arith.constant 6 : index
  %gb = lego.group_by [%c6](%ob1, %ob2) : !lego.layout
  %f = lego.apply %gb(%i) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_groupby_empty
// CHECK:       lego.group_by[%{{.*}}] () : !lego.layout
func.func @parse_groupby_empty(%i: index) -> index {
  %c10 = arith.constant 10 : index
  %gb = lego.group_by [%c10]() : !lego.layout
  %f = lego.apply %gb(%i) : !lego.layout
  return %f : index
}

// ============================================================================
// TileByOp
// ============================================================================

// CHECK-LABEL: func.func @lego_tile_by
// CHECK:       lego.tile_by %{{.*}} tile_dims {{\[}}[%{{.*}}], [%{{.*}}]{{\]}} : !lego.layout
func.func @lego_tile_by(%inner: !lego.layout) -> !lego.layout {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %tb = lego.tile_by %inner tile_dims [[%c4], [%c8]] : !lego.layout
  return %tb : !lego.layout
}

// CHECK-LABEL: func.func @lego_tile_by_complex
// CHECK:       lego.tile_by %{{.*}} tile_dims {{\[}}[%{{.*}}, %{{.*}}, %{{.*}}]{{\]}} : !lego.layout
func.func @lego_tile_by_complex(%inner: !lego.layout) -> !lego.layout {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %tb = lego.tile_by %inner tile_dims [[%c2, %c4, %c8]] : !lego.layout
  return %tb : !lego.layout
}

// CHECK-LABEL: func.func @lego_tile_by_nested
// CHECK:       lego.tile_by %{{.*}} tile_dims {{\[}}[%{{.*}}], [%{{.*}}]{{\]}} : !lego.layout
func.func @lego_tile_by_nested(%inner: !lego.layout) -> !lego.layout {
  %c4 = arith.constant 4 : index
  %tb = lego.tile_by %inner tile_dims [[%c4], [%c4]] : !lego.layout
  return %tb : !lego.layout
}

// ============================================================================
// ApplyOp / ApplyInverseOp
// ============================================================================

// CHECK-LABEL: func.func @parse_apply
// CHECK:       lego.apply %{{.*}}(%{{.*}}, %{{.*}}) : !lego.layout
func.func @parse_apply(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %r = lego.row [%c4, %c8] : !lego.layout
  %f = lego.apply %r(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_apply_inverse
// CHECK:       lego.apply_inverse %{{.*}}(%{{.*}}) : !lego.layout -> index, index
func.func @parse_apply_inverse(%f: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %r = lego.row [%c4, %c8] : !lego.layout
  %i, %j = lego.apply_inverse %r(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// CHECK-LABEL: func.func @parse_apply_inverse_3d
// CHECK:       lego.apply_inverse %{{.*}}(%{{.*}}) : !lego.layout -> index, index, index
func.func @parse_apply_inverse_3d(%f: index) -> (index, index, index) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %r = lego.row [%c2, %c3, %c4] : !lego.layout
  %i, %j, %k = lego.apply_inverse %r(%f) : !lego.layout -> index, index, index
  return %i, %j, %k : index, index, index
}

// ============================================================================
// Nested / Composed ops (parse only)
// ============================================================================

// CHECK-LABEL: func.func @parse_nested_orderby
// CHECK:       lego.order_by(%{{.*}}) : !lego.layout
// CHECK:       lego.order_by(%{{.*}}, %{{.*}}) : !lego.layout
func.func @parse_nested_orderby(%i: index, %j: index, %k: index) -> index {
  %c2 = arith.constant 2 : index
  %p1 = lego.reg_p perm [0] dims [%c2] : !lego.layout
  %c3 = arith.constant 3 : index
  %p2 = lego.reg_p perm [0] dims [%c3] : !lego.layout
  %inner = lego.order_by(%p1, %p2) : !lego.layout
  %c4 = arith.constant 4 : index
  %p3 = lego.reg_p perm [0] dims [%c4] : !lego.layout
  %outer = lego.order_by(%inner, %p3) : !lego.layout
  %f = lego.apply %outer(%i, %j, %k) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_groupby_with_orderby_chain
// CHECK:       lego.order_by(%{{.*}}, %{{.*}}) : !lego.layout
// CHECK:       lego.group_by[%{{.*}}, %{{.*}}] (%{{.*}}) : !lego.layout
func.func @parse_groupby_with_orderby_chain(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %p1 = lego.reg_p perm [0] dims [%c4] : !lego.layout
  %c8 = arith.constant 8 : index
  %p2 = lego.reg_p perm [0] dims [%c8] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %gb = lego.group_by [%c4, %c8](%ob) : !lego.layout
  %f = lego.apply %gb(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_tileby_with_regp_inner
// CHECK:       lego.reg_p perm [1, 0] dims[%{{.*}}, %{{.*}}] : !lego.layout
// CHECK:       lego.tile_by %{{.*}} tile_dims {{\[}}[%{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}]{{\]}} : !lego.layout
func.func @parse_tileby_with_regp_inner(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %inner = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  // TileBy inner OrderBy?
  // Our implementation assumes input is OrderBy or single block.
  // RegP is acceptable.
  %c2 = arith.constant 2 : index
  %tb = lego.tile_by %inner tile_dims [[%c2, %c2], [%c2, %c4]] : !lego.layout
  %f = lego.apply %tb(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @parse_multiple_layouts
// CHECK:       lego.row[%{{.*}}, %{{.*}}] : !lego.layout
// CHECK:       lego.col[%{{.*}}, %{{.*}}] : !lego.layout
// CHECK:       lego.apply %{{.*}}(%{{.*}}, %{{.*}}) : !lego.layout
// CHECK:       lego.apply %{{.*}}(%{{.*}}, %{{.*}}) : !lego.layout
func.func @parse_multiple_layouts(%i: index, %j: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %row = lego.row [%c4, %c8] : !lego.layout
  %col = lego.col [%c4, %c8] : !lego.layout
  %frow = lego.apply %row(%i, %j) : !lego.layout
  %fcol = lego.apply %col(%i, %j) : !lego.layout
  return %frow, %fcol : index, index
}

// CHECK-LABEL: func.func @parse_layout_reuse
// CHECK:       lego.row[%{{.*}}, %{{.*}}] : !lego.layout
// CHECK:       lego.apply %{{.*}}(%{{.*}}, %{{.*}}) : !lego.layout
// CHECK:       lego.apply %{{.*}}(%{{.*}}, %{{.*}}) : !lego.layout
func.func @parse_layout_reuse(%i1: index, %j1: index, %i2: index, %j2: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %r = lego.row [%c4, %c8] : !lego.layout
  %f1 = lego.apply %r(%i1, %j1) : !lego.layout
  %f2 = lego.apply %r(%i2, %j2) : !lego.layout
  return %f1, %f2 : index, index
}
