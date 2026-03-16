// RUN: lego-opt %s -lego-lower -split-input-file | FileCheck %s
//
// ============================================================================
// Matmul kernel layout tests (symbolic dimensions)
//
// Verifies that the MLIR lowering produces clean, simple arithmetic —
// no bloated div/rem chains — for the standard matmul tiling patterns.
// ============================================================================

// -----

// L_A = OrderBy(Row(M, K)).TileBy([M/BM, K/BK], [BM, BK])
// Expected: (pid_m*BM + i)*K + k*BK + j
//   = pid_m*BM*K + i*K + k*BK + j  (no div/rem)

// CHECK-LABEL: func.func @matmul_a_ptr
// CHECK-SAME:  (%[[BK:.*]]: index, %[[BM:.*]]: index, %[[K:.*]]: index, %[[M:.*]]: index, %[[PM:.*]]: index, %[[PK:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index)
// CHECK:       %[[T0:.*]] = arith.muli %[[PM]], %[[BM]] : index
// CHECK:       %[[C0:.*]] = arith.addi %[[I]], %[[T0]] : index
// CHECK:       %[[T1:.*]] = arith.muli %[[PK]], %[[BK]] : index
// CHECK:       %[[C1:.*]] = arith.addi %[[J]], %[[T1]] : index
// CHECK:       %[[S:.*]] = arith.muli %[[C0]], %[[K]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[C1]], %[[S]] : index
// CHECK:       return %[[RES]] : index
// CHECK-NOT:   arith.divui
// CHECK-NOT:   arith.remui
func.func @matmul_a_ptr(%BK: index, %BM: index, %K: index, %M: index,
                         %pid_m: index, %k: index, %i: index, %j: index) -> index {
  %c1 = arith.constant 1 : index
  lego.assume_bounds %BK lb: %c1
  lego.assume_bounds %BM lb: %c1
  lego.assume_bounds %K lb: %c1
  lego.assume_bounds %M lb: %c1
  %c0 = arith.constant 0 : index
  %D0 = arith.divui %M, %BM : index
  %D1 = arith.divui %K, %BK : index
  lego.assume_bounds %pid_m lb: %c0 ub: %D0
  lego.assume_bounds %k lb: %c0 ub: %D1
  lego.assume_bounds %i lb: %c0 ub: %BM
  lego.assume_bounds %j lb: %c0 ub: %BK
  %row = lego.reg_p perm [0, 1] dims[%M, %K] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[%D0, %D1], [%BM, %BK]] : !lego.layout
  %f = lego.apply %tiled(%pid_m, %k, %i, %j) : !lego.layout
  return %f : index
}

// -----

// L_C = OrderBy(Row(M, N)).TileBy([M/BM, N/BN], [BM, BN])
// Expected: (pid_m*BM + i)*N + pid_n*BN + j

// CHECK-LABEL: func.func @matmul_c_ptr
// CHECK-SAME:  (%[[BM:.*]]: index, %[[BN:.*]]: index, %[[M:.*]]: index, %[[N:.*]]: index, %[[PM:.*]]: index, %[[PN:.*]]: index, %[[I:.*]]: index, %[[J:.*]]: index)
// CHECK:       %[[T0:.*]] = arith.muli %[[PM]], %[[BM]] : index
// CHECK:       %[[C0:.*]] = arith.addi %[[I]], %[[T0]] : index
// CHECK:       %[[T1:.*]] = arith.muli %[[PN]], %[[BN]] : index
// CHECK:       %[[C1:.*]] = arith.addi %[[J]], %[[T1]] : index
// CHECK:       %[[S:.*]] = arith.muli %[[C0]], %[[N]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[C1]], %[[S]] : index
// CHECK:       return %[[RES]] : index
// CHECK-NOT:   arith.divui
// CHECK-NOT:   arith.remui
func.func @matmul_c_ptr(%BM: index, %BN: index, %M: index, %N: index,
                         %pid_m: index, %pid_n: index, %i: index, %j: index) -> index {
  %c1 = arith.constant 1 : index
  lego.assume_bounds %BM lb: %c1
  lego.assume_bounds %BN lb: %c1
  lego.assume_bounds %M lb: %c1
  lego.assume_bounds %N lb: %c1
  %c0 = arith.constant 0 : index
  %D0 = arith.divui %M, %BM : index
  %D1 = arith.divui %N, %BN : index
  lego.assume_bounds %pid_m lb: %c0 ub: %D0
  lego.assume_bounds %pid_n lb: %c0 ub: %D1
  lego.assume_bounds %i lb: %c0 ub: %BM
  lego.assume_bounds %j lb: %c0 ub: %BN
  %row = lego.reg_p perm [0, 1] dims[%M, %N] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[%D0, %D1], [%BM, %BN]] : !lego.layout
  %f = lego.apply %tiled(%pid_m, %pid_n, %i, %j) : !lego.layout
  return %f : index
}

// -----

// L_pid = OrderBy(Col(max(npm/GM,1), 1), Col(min(GM,npm), npn)).TileBy([npm, npn])
// pid_m = group_id * min_gm + pid % min_gm
// pid_n = (pid % (min_gm * npn)) / min_gm
//
// Symbolically matches the reference Triton matmul kernel's PID remapping.

// CHECK-LABEL: func.func @matmul_pid_inv
// CHECK-SAME:  (%[[GM:.*]]: index, %[[NPM:.*]]: index, %[[NPN:.*]]: index, %[[PID:.*]]: index)
// CHECK:       %[[CMP:.*]] = arith.cmpi sle, %[[GM]], %[[NPM]] : index
// CHECK:       %[[MINGM:.*]] = arith.select %[[CMP]], %[[GM]], %[[NPM]] : index
// CHECK:       %[[GS:.*]] = arith.muli %[[MINGM]], %[[NPN]] : index
// CHECK:       %[[LOCAL:.*]] = arith.remui %[[PID]], %[[GS]] : index
// CHECK:       %[[GID:.*]] = arith.divui %[[PID]], %[[GS]] : index
// CHECK:       %[[PN:.*]] = arith.divui %[[LOCAL]], %[[MINGM]] : index
// CHECK:       %[[LM:.*]] = arith.remui %[[LOCAL]], %[[MINGM]] : index
// CHECK:       %[[GO:.*]] = arith.muli %[[GID]], %[[MINGM]] : index
// CHECK:       %[[PM:.*]] = arith.addi %[[LM]], %[[GO]] : index
// CHECK:       return %[[PM]], %[[PN]] : index, index
func.func @matmul_pid_inv(%GM: index, %npm: index, %npn: index, %pid: index)
    -> (index, index) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  lego.assume_bounds %GM lb: %c1
  lego.assume_bounds %npm lb: %c1
  lego.assume_bounds %npn lb: %c1
  %total = arith.muli %npm, %npn : index
  lego.assume_bounds %pid lb: %c0 ub: %total
  %G = arith.divui %npm, %GM : index
  %cmp1 = arith.cmpi sle, %G, %c1 : index
  %Gsel = arith.select %cmp1, %c1, %G : index
  %cmp2 = arith.cmpi sle, %GM, %npm : index
  %mingm = arith.select %cmp2, %GM, %npm : index
  %col0 = lego.reg_p perm [1, 0] dims[%Gsel, %c1] : !lego.layout
  %col1 = lego.reg_p perm [1, 0] dims[%mingm, %npn] : !lego.layout
  %ob = lego.order_by(%col0, %col1) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[%npm, %npn]] : !lego.layout
  %r:2 = lego.apply_inverse %tiled(%pid) : !lego.layout -> index, index
  return %r#0, %r#1 : index, index
}
