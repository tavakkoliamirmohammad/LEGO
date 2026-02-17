// RUN: lego-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @apply_identity
func.func @apply_identity(%layout: !lego.layout, %flat: index) -> index {
  // CHECK-NOT: lego.apply_inverse
  // CHECK-NOT: lego.apply
  // CHECK: return %arg1
  %indices:2 = lego.apply_inverse %layout(%flat) : !lego.layout -> index, index
  %res = lego.apply %layout(%indices#0, %indices#1) : !lego.layout
  return %res : index
}

// CHECK-LABEL: @apply_inverse_identity
func.func @apply_inverse_identity(%layout: !lego.layout, %i: index, %j: index) -> (index, index) {
  // CHECK-NOT: lego.apply
  // CHECK-NOT: lego.apply_inverse
  // CHECK: return %arg1, %arg2
  %flat = lego.apply %layout(%i, %j) : !lego.layout
  %res:2 = lego.apply_inverse %layout(%flat) : !lego.layout -> index, index
  return %res#0, %res#1 : index, index
}

// CHECK-LABEL: @simplify_gen_p_identity
func.func @simplify_gen_p_identity() -> !lego.layout {
  // CHECK: %[[REG:.*]] = lego.reg_p perm [0] dims [16]
  // CHECK: return %[[REG]]
  %layout = lego.gen_p [16]
    apply (%i: index) {
      lego.yield %i : index
    }
    inv (%flat: index) {
      lego.yield %flat : index
    } : !lego.layout
  return %layout : !lego.layout
}

// CHECK-LABEL: @simplify_gen_p_strided
func.func @simplify_gen_p_strided() -> !lego.layout {
  // dims [4, 8]. flat = j * 4 + i  => perm [1, 0]
  // CHECK: %[[REG:.*]] = lego.reg_p perm [1, 0] dims [4, 8]
  // CHECK: return %[[REG]]
  %layout = lego.gen_p [4, 8]
    apply (%i: index, %j: index) {
      %c4 = arith.constant 4 : index
      %m = arith.muli %j, %c4 : index
      %sum = arith.addi %m, %i : index
      lego.yield %sum : index
    }
    inv (%flat: index) {
      %c4 = arith.constant 4 : index
      %i = arith.remui %flat, %c4 : index
      %j = arith.divui %flat, %c4 : index
      lego.yield %i, %j : index, index
    } : !lego.layout
  return %layout : !lego.layout
}
