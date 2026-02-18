// RUN: lego-opt --lego-arith-simplification --int-range-optimizations --canonicalize %s | FileCheck %s

// A1: (d*q + r) % d -> r % d
// CHECK-LABEL: func.func @test_A1
// CHECK-SAME: (%[[Q:.*]]: index, %[[R:.*]]: index)
func.func @test_A1(%q: index, %r: index) -> index {
  %c10 = arith.constant 10 : index
  %0 = arith.muli %q, %c10 : index
  %1 = arith.addi %0, %r : index
  
  // CHECK: %[[REM:.*]] = arith.remui %[[R]], %{{.*}}
  // CHECK: return %[[REM]]
  %2 = arith.remui %1, %c10 : index
  return %2 : index
}

// A2: (d*q + r) / d -> q (if r < d)
// CHECK-LABEL: func.func @test_A2
// CHECK-SAME: (%[[Q:.*]]: index, %[[R:.*]]: index)
func.func @test_A2(%q: index, %r: index) -> index {
  %c10 = arith.constant 10 : index
  // Ensure r < 10 by using remui
  %r_limited = arith.remui %r, %c10 : index
  
  %0 = arith.muli %q, %c10 : index
  %1 = arith.addi %0, %r_limited : index
  
  // CHECK: return %[[Q]]
  %2 = arith.divui %1, %c10 : index
  return %2 : index
}

// A3: (x % d) / d -> 0
// CHECK-LABEL: func.func @test_A3
// CHECK-SAME: (%[[X:.*]]: index)
func.func @test_A3(%x: index) -> index {
  %c10 = arith.constant 10 : index
  %0 = arith.remui %x, %c10 : index
  
  // CHECK: %[[ZERO:.*]] = arith.constant 0
  // CHECK: return %[[ZERO]]
  %1 = arith.divui %0, %c10 : index
  return %1 : index
}

// B/C: (x / a) * a + (x % a) -> x
// CHECK-LABEL: func.func @test_B_C
// CHECK-SAME: (%[[X:.*]]: index)
func.func @test_B_C(%x: index) -> index {
  %c10 = arith.constant 10 : index
  %div = arith.divui %x, %c10 : index
  %rem = arith.remui %x, %c10 : index
  %mul = arith.muli %div, %c10 : index
  %res = arith.addi %mul, %rem : index
  
  // CHECK: return %[[X]]
  return %res : index
}

  // A6: (c*d + r) / d -> c + r/d (Constant handling)
  // CHECK-LABEL: func.func @test_A6_const
  // CHECK-SAME: (%[[R:.*]]: index)
  func.func @test_A6_const(%r: index) -> index {
    %c10 = arith.constant 10 : index
    %c20 = arith.constant 20 : index
    %0 = arith.addi %r, %c20 : index
    
    // Expect: 2 + r/10
    // CHECK: %[[C2:.*]] = arith.constant 2
    // CHECK: %[[DIV:.*]] = arith.divui %[[R]], %{{.*}}
    // CHECK: %[[RES:.*]] = arith.addi %[[DIV]], %[[C2]]
    // CHECK: return %[[RES]]
    %1 = arith.divui %0, %c10 : index
    return %1 : index
  }

  // A2 check: (q*d + r) / d with r potentially >= d
  // CHECK-LABEL: func.func @test_A2_large_rem
  // CHECK-SAME: (%[[Q:.*]]: index, %[[R:.*]]: index)
  func.func @test_A2_large_rem(%q: index, %r: index) -> index {
    %c10 = arith.constant 10 : index
    %0 = arith.muli %q, %c10 : index
    %1 = arith.addi %0, %r : index
    
    // Should rewrite to q + r/10
    // CHECK: %[[DIV_REM:.*]] = arith.divui %[[R]], %{{.*}}
    // CHECK: %[[RES:.*]] = arith.addi %[[Q]], %[[DIV_REM]]
    // CHECK: return %[[RES]]
    %2 = arith.divui %1, %c10 : index
    return %2 : index
  }
