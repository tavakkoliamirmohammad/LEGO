// RUN: lego-opt %s -lego-normalization | FileCheck %s

// CHECK-LABEL: func @test_row_normalization
func.func @test_row_normalization() -> !lego.layout {
  // Row(10) -> RegP([10], [0])
  %c10 = arith.constant 10 : index
  %0 = lego.row [%c10] : !lego.layout
  return %0 : !lego.layout
}
// CHECK: %[[REGP:.*]] = lego.reg_p perm [0] dims[%{{.*}}]
// CHECK: return %[[REGP]]

// CHECK-LABEL: func @test_col_normalization
func.func @test_col_normalization() -> !lego.layout {
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
// CHECK: %[[OB:.*]] = lego.order_by(%[[R_REGP]], %[[C_REGP]])
// CHECK: %[[RP1:.*]] = lego.reg_p perm [0, 1] dims[%{{.*}}, %{{.*}}]
// CHECK: %[[OB1:.*]] = lego.order_by(%[[RP1]])
// CHECK: %[[RPF:.*]] = lego.reg_p perm [0, 1, 2, 3] dims[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}]
// CHECK: %[[OBF:.*]] = lego.order_by(%[[RPF]])
// CHECK: lego.group_by[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] (%[[OB]], %[[OB1]], %[[OBF]])

// CHECK-LABEL: func @test_tile_by_d2_q3
func.func @test_tile_by_d2_q3() -> !lego.layout {
  // TileBy(Row(6, 6), [[2, 2], [3, 3], [1, 1]])
  // d=2, q=3 → σ=[0,2,4,1,3,5], σ⁻¹=[0,3,1,4,2,5]
  %c6 = arith.constant 6 : index
  %0 = lego.row [%c6, %c6] : !lego.layout
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %1 = lego.tile_by %0 tile_dims [[%c2, %c2], [%c3, %c3], [%c1, %c1]] : !lego.layout
  return %1 : !lego.layout
}

// Middle reshuffle for Row(6,6): d=2, q=1 → σ=[0,1] (identity)
// CHECK: %[[ROW_REGP:.*]] = lego.reg_p perm [0, 1] dims[%{{.*}}, %{{.*}}]
// CHECK: %[[RP_MID:.*]] = lego.reg_p perm [0, 1] dims[%{{.*}}, %{{.*}}]
// CHECK: %[[OB_MID:.*]] = lego.order_by(%[[RP_MID]])
// Final reshuffle: d=2, q=3 → σ⁻¹=[0, 3, 1, 4, 2, 5]
// CHECK: %[[RPF:.*]] = lego.reg_p perm [0, 3, 1, 4, 2, 5] dims[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}]
// CHECK: %[[OBF:.*]] = lego.order_by(%[[RPF]])
// CHECK: lego.group_by[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] (%[[ROW_REGP]], %[[OB_MID]], %[[OBF]])

// CHECK-LABEL: func.func @test_assume_bounds
// CHECK-SAME: (%[[VAL:arg[0-9]+]]: index, %[[LB:arg[0-9]+]]: index, %[[UB:arg[0-9]+]]: index)
func.func @test_assume_bounds(%arg0: index, %arg1: index, %arg2: index) {
  lego.assume_bounds %arg0 lb: %arg1 ub: %arg2
  return
}
// CHECK: %[[CMP_LB:.*]] = arith.cmpi sge, %[[VAL]], %[[LB]] : index
// CHECK: lego.assume %[[CMP_LB]]
// CHECK: %[[CMP_UB:.*]] = arith.cmpi slt, %[[VAL]], %[[UB]] : index
// CHECK: lego.assume %[[CMP_UB]]
// CHECK: return

// CHECK-LABEL: func.func @test_assume_bounds_lb_only
// CHECK-SAME: (%[[VAL:arg[0-9]+]]: index, %[[LB:arg[0-9]+]]: index)
func.func @test_assume_bounds_lb_only(%arg0: index, %arg1: index) {
  lego.assume_bounds %arg0 lb: %arg1
  return
}
// CHECK: %[[CMP_LB:.*]] = arith.cmpi sge, %[[VAL]], %[[LB]] : index
// CHECK: lego.assume %[[CMP_LB]]
// CHECK-NOT: arith.cmpi slt
// CHECK: return

// CHECK-LABEL: func.func @test_assume_bounds_ub_only
// CHECK-SAME: (%[[VAL:arg[0-9]+]]: index, %[[UB:arg[0-9]+]]: index)
func.func @test_assume_bounds_ub_only(%arg0: index, %arg1: index) {
  lego.assume_bounds %arg0 ub: %arg1
  return
}
// CHECK: %[[CMP_UB:.*]] = arith.cmpi slt, %[[VAL]], %[[UB]] : index
// CHECK: lego.assume %[[CMP_UB]]
// CHECK-NOT: arith.cmpi sge
// CHECK: return
