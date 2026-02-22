// RUN: lego-opt %s -lego-desugar | FileCheck %s

// CHECK-LABEL: func @test_row_desugar
func.func @test_row_desugar() -> !lego.layout {
  // Row(10) -> RegP([10], [0])
  %c10 = arith.constant 10 : index
  %0 = lego.row [%c10] : !lego.layout
  return %0 : !lego.layout
}
// CHECK: %[[REGP:.*]] = lego.reg_p perm [0] dims[%{{.*}}]
// CHECK: return %[[REGP]]

// CHECK-LABEL: func @test_col_desugar
func.func @test_col_desugar() -> !lego.layout {
  // Col(10, 20) -> RegP([10, 20], [1, 0])
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %0 = lego.col [%c10, %c20] : !lego.layout
  return %0 : !lego.layout
}
// CHECK: %[[REGP:.*]] = lego.reg_p perm [1, 0] dims[%{{.*}}, %{{.*}}]
// CHECK: return %[[REGP]]

// CHECK-LABEL: func @test_tile_by_row
func.func @test_tile_by_row() -> !lego.layout {
  // TileBy(Row(12, 12), [[4, 4], [3, 3]])
  %c12 = arith.constant 12 : index
  %0 = lego.row [%c12, %c12] : !lego.layout
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %1 = lego.tile_by %0 tile_dims [[%c4, %c4], [%c3, %c3]] : !lego.layout
  return %1 : !lego.layout
}

// CHECK: %[[ROW_REGP:.*]] = lego.reg_p perm [0, 1] dims[%{{.*}}, %{{.*}}]
// CHECK: %[[REGP1:.*]] = lego.reg_p perm [0, 1] dims[%{{.*}}, %{{.*}}]
// CHECK: %[[OB1:.*]] = lego.order_by(%[[REGP1]])
// CHECK: %[[REGP2:.*]] = lego.reg_p perm [0, 2, 1, 3] dims[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}]
// CHECK: %[[OB2:.*]] = lego.order_by(%[[REGP2]])
// CHECK: %[[RES:.*]] = lego.group_by[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] (%[[ROW_REGP]], %[[OB1]], %[[OB2]])
// CHECK: return %[[RES]]

// CHECK-LABEL: func @test_tile_by_chain
func.func @test_tile_by_chain() -> !lego.layout {
  // Chain: OrderBy(Row(10), Col(20))
  %c10 = arith.constant 10 : index
  %r = lego.row [%c10] : !lego.layout
  %c20 = arith.constant 20 : index
  %c = lego.col [%c20] : !lego.layout
  %ob = lego.order_by(%r, %c) : !lego.layout
  
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %c4 = arith.constant 4 : index
  %tb = lego.tile_by %ob tile_dims [[%c2], [%c5], [%c5], [%c4]] : !lego.layout
  return %tb : !lego.layout
}

// CHECK: %[[R_REGP:.*]] = lego.reg_p perm [0] dims[%{{.*}}]
// CHECK: %[[C_REGP:.*]] = lego.reg_p perm [0] dims[%{{.*}}]
// CHECK: %[[RP1:.*]] = lego.reg_p perm [0] dims[%{{.*}}]
// CHECK: %[[OB1:.*]] = lego.order_by(%[[RP1]])
// CHECK: %[[RP2:.*]] = lego.reg_p perm [0] dims[%{{.*}}]
// CHECK: %[[OB2:.*]] = lego.order_by(%[[RP2]])
// CHECK: %[[RPF:.*]] = lego.reg_p perm [0, 1, 2, 3] dims[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}]
// CHECK: %[[OBF:.*]] = lego.order_by(%[[RPF]])
// CHECK: lego.group_by[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] (%[[R_REGP]], %[[OB1]], %[[C_REGP]], %[[OB2]], %[[OBF]])
