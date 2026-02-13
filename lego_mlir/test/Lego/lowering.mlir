// RUN: lego-opt %s -lego-to-arith | FileCheck %s

// CHECK-LABEL: func @test_row_apply
func.func @test_row_apply(%i: index, %j: index) -> index {
  // CHECK: %[[M:.*]] = arith.constant 10 : index
  // CHECK: %[[IM:.*]] = arith.muli %arg0, %[[M]] : index
  // CHECK: %[[RES:.*]] = arith.addi %[[IM]], %arg1 : index
  // CHECK: return %[[RES]] : index
  %row = lego.row 10, 10 : !lego.layout
  %flat = lego.apply %row(%i, %j) : !lego.layout
  return %flat : index
}

// CHECK-LABEL: func @test_row_inv
func.func @test_row_inv(%flat: index) -> (index, index) {
  // CHECK: %[[M:.*]] = arith.constant 10 : index
  // CHECK: %[[I:.*]] = arith.divui %arg0, %[[M]] : index
  // CHECK: %[[J:.*]] = arith.remui %arg0, %[[M]] : index
  // CHECK: return %[[I]], %[[J]] : index, index
  %row = lego.row 10, 10 : !lego.layout
  %i, %j = lego.apply_inverse %row(%flat) : !lego.layout -> (index, index)
  return %i, %j : index, index
}

// CHECK-LABEL: func @test_regp_apply
func.func @test_regp_apply(%i: index, %j: index) -> index {
  // perm [1, 0], dims [10, 20] -> permuted dims [20, 10]
  // indices (i, j) -> (j, i)
  // flat = j * 10 + i
  // CHECK: %[[DIM1:.*]] = arith.constant 10 : index
  // CHECK: %[[TERM:.*]] = arith.muli %arg1, %[[DIM1]] : index
  // CHECK: %[[RES:.*]] = arith.addi %[[TERM]], %arg0 : index
  // CHECK: return %[[RES]] : index
  %regp = lego.reg_p perm [1, 0] dims [10, 20] : !lego.layout
  %flat = lego.apply %regp(%i, %j) : !lego.layout
  return %flat : index
}
