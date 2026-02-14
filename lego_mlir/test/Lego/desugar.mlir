// RUN: lego-opt %s -lego-desugar | FileCheck %s

// CHECK-LABEL: func @test_row_desugar
func.func @test_row_desugar() -> !lego.layout {
  // Row(10) -> RegP([10], [0])
  %0 = lego.row [10] : !lego.layout
  return %0 : !lego.layout
}
// CHECK: %[[REGP:.*]] = lego.reg_p perm [0] dims [10]
// CHECK: return %[[REGP]]

// CHECK-LABEL: func @test_col_desugar
func.func @test_col_desugar() -> !lego.layout {
  // Col(10, 20) -> RegP([10, 20], [1, 0])
  %0 = lego.col [10, 20] : !lego.layout
  return %0 : !lego.layout
}
// CHECK: %[[REGP:.*]] = lego.reg_p perm [1, 0] dims [10, 20]
// CHECK: return %[[REGP]]

// CHECK-LABEL: func @test_tile_by_row
func.func @test_tile_by_row() -> !lego.layout {
  // TileBy(Row(12, 12), [[4, 4], [3, 3]])
  %0 = lego.row [12, 12] : !lego.layout
  %1 = lego.tile_by %0 tile_dims [[4, 4], [3, 3]] : !lego.layout
  return %1 : !lego.layout
}

// CHECK: %[[ROW_REGP:.*]] = lego.reg_p perm [0, 1] dims [12, 12]
// CHECK: %[[REGP1:.*]] = lego.reg_p perm [0, 1] dims [12, 12]
// CHECK: %[[OB1:.*]] = lego.order_by(%[[REGP1]])
// CHECK: %[[REGP2:.*]] = lego.reg_p perm [0, 2, 1, 3] dims [4, 4, 3, 3]
// CHECK: %[[OB2:.*]] = lego.order_by(%[[REGP2]])
// CHECK: %[[RES:.*]] = lego.group_by [4, 4, 3, 3](%[[ROW_REGP]], %[[OB1]], %[[OB2]])
// CHECK: return %[[RES]]

// CHECK-LABEL: func @test_tile_by_chain
func.func @test_tile_by_chain() -> !lego.layout {
  // Chain: OrderBy(Row(10), Col(20))
  %r = lego.row [10] : !lego.layout
  %c = lego.col [20] : !lego.layout
  %ob = lego.order_by(%r, %c) : !lego.layout
  
  %tb = lego.tile_by %ob tile_dims [[2], [5], [5], [4]] : !lego.layout
  return %tb : !lego.layout
}

// CHECK: %[[R_REGP:.*]] = lego.reg_p perm [0] dims [10]
// CHECK: %[[C_REGP:.*]] = lego.reg_p perm [0] dims [20]
// CHECK: %[[RP1:.*]] = lego.reg_p perm [0] dims [10]
// CHECK: %[[OB1:.*]] = lego.order_by(%[[RP1]])
// CHECK: %[[RP2:.*]] = lego.reg_p perm [0] dims [20]
// CHECK: %[[OB2:.*]] = lego.order_by(%[[RP2]])
// CHECK: %[[RPF:.*]] = lego.reg_p perm [0, 1, 2, 3] dims [2, 5, 5, 4]
// CHECK: %[[OBF:.*]] = lego.order_by(%[[RPF]])
// CHECK: lego.group_by [2, 5, 5, 4](%[[R_REGP]], %[[OB1]], %[[C_REGP]], %[[OB2]], %[[OBF]])
