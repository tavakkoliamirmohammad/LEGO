// RUN: lego-opt --int-range-optimizations --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @simplify_rem
// CHECK-SAME: (%[[ARG0:.*]]: index)
func.func @simplify_rem(%arg0: index) -> index {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  
  // %0 is in range [0, 9]
  // CHECK: %[[R1:.*]] = arith.remui %[[ARG0]], %{{.*}}
  %0 = arith.remui %arg0, %c10 : index
  
  // %0 % 20 should be %0 because max(%0) < 20
  // CHECK-NOT: arith.remui
  // CHECK: return %[[R1]]
  %1 = arith.remui %0, %c20 : index
  
  return %1 : index
}

// CHECK-LABEL: func.func @simplify_div
// CHECK-SAME: (%[[ARG0:.*]]: index)
func.func @simplify_div(%arg0: index) -> index {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  
  // %0 is in range [0, 9]
  %0 = arith.remui %arg0, %c10 : index
  
  // %0 / 20 should be 0 because max(%0) < 20
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK: return %[[ZERO]]
  %1 = arith.divui %0, %c20 : index
  
  return %1 : index
}

// CHECK-LABEL: func.func @simplify_chained_rem
// CHECK-SAME: (%[[ARG0:.*]]: index)
func.func @simplify_chained_rem(%arg0: index) -> index {
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index

  // %0 in [0, 4]
  // CHECK: %[[R1:.*]] = arith.remui %[[ARG0]], %{{.*}}
  %0 = arith.remui %arg0, %c5 : index
  
  // %0 % 10 -> %0
  %1 = arith.remui %0, %c10 : index
  
  // %1 % 20 -> %1
  %2 = arith.remui %1, %c20 : index

  // CHECK-NOT: arith.remui
  // CHECK: return %[[R1]]
  return %2 : index
}

// CHECK-LABEL: func.func @simplify_add_rem
// CHECK-SAME: (%[[ARG0:.*]]: index)
func.func @simplify_add_rem(%arg0: index) -> index {
  %c5 = arith.constant 5 : index
  %c8 = arith.constant 8 : index
  %c20 = arith.constant 20 : index

  // CHECK: %[[R1:.*]] = arith.remui %[[ARG0]], %{{.*}}
  %0 = arith.remui %arg0, %c5 : index
  
  // CHECK: %[[R2:.*]] = arith.remui %[[ARG0]], %{{.*}}
  %1 = arith.remui %arg0, %c8 : index
  
  // %2 in [0, 11] (max 4 + 7 = 11)
  // CHECK: %[[SUM:.*]] = arith.addi %[[R1]], %[[R2]]
  %2 = arith.addi %0, %1 : index

  // 11 < 20, so %2 % 20 -> %2
  // CHECK-NOT: arith.remui
  // CHECK: return %[[SUM]]
  %3 = arith.remui %2, %c20 : index

  return %3 : index
}
