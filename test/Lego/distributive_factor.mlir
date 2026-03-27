// RUN: lego-opt %s -lego-arith-simplification | FileCheck %s
//
// Tests for the DistributiveFactor pattern:
//   muli(a, c) + muli(b, c)  →  muli(addi(a, b), c)

// CHECK-LABEL: func.func @factor_shared_rhs
// CHECK-SAME:  (%[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index)
// CHECK:       %[[SUM:.*]] = arith.addi %[[A]], %[[B]] : index
// CHECK:       %[[RES:.*]] = arith.muli %[[SUM]], %[[C]] : index
// CHECK:       return %[[RES]] : index
func.func @factor_shared_rhs(%a: index, %b: index, %c: index) -> index {
  %ac = arith.muli %a, %c : index
  %bc = arith.muli %b, %c : index
  %res = arith.addi %ac, %bc : index
  return %res : index
}

// Commutative: c*a + c*b → (a+b)*c
// CHECK-LABEL: func.func @factor_shared_lhs
// CHECK-SAME:  (%[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index)
// CHECK:       %[[SUM:.*]] = arith.addi %[[A]], %[[B]] : index
// CHECK:       %[[RES:.*]] = arith.muli %[[SUM]], %[[C]] : index
// CHECK:       return %[[RES]] : index
func.func @factor_shared_lhs(%a: index, %b: index, %c: index) -> index {
  %ca = arith.muli %c, %a : index
  %cb = arith.muli %c, %b : index
  %res = arith.addi %ca, %cb : index
  return %res : index
}

// Mixed: a*c + c*b → (a+b)*c
// CHECK-LABEL: func.func @factor_mixed_commute
// CHECK-SAME:  (%[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index)
// CHECK:       %[[SUM:.*]] = arith.addi %[[A]], %[[B]] : index
// CHECK:       %[[RES:.*]] = arith.muli %[[SUM]], %[[C]] : index
// CHECK:       return %[[RES]] : index
func.func @factor_mixed_commute(%a: index, %b: index, %c: index) -> index {
  %ac = arith.muli %a, %c : index
  %cb = arith.muli %c, %b : index
  %res = arith.addi %ac, %cb : index
  return %res : index
}

// No shared factor — should not fire.
// CHECK-LABEL: func.func @no_shared_factor
// CHECK:       arith.muli
// CHECK:       arith.muli
// CHECK:       arith.addi
func.func @no_shared_factor(%a: index, %b: index, %c: index, %d: index) -> index {
  %ac = arith.muli %a, %c : index
  %bd = arith.muli %b, %d : index
  %res = arith.addi %ac, %bd : index
  return %res : index
}

// Three-term chain: a*x + b*x + c*x → (a+b+c)*x via two applications
// CHECK-LABEL: func.func @three_term_factor
// CHECK-SAME:  (%[[A:.*]]: index, %[[B:.*]]: index, %[[C:.*]]: index, %[[X:.*]]: index)
// CHECK:       %[[SUM1:.*]] = arith.addi
// CHECK:       %[[SUM2:.*]] = arith.addi
// CHECK:       %[[RES:.*]] = arith.muli %[[SUM2]], %[[X]] : index
// CHECK:       return %[[RES]] : index
func.func @three_term_factor(%a: index, %b: index, %c: index, %x: index) -> index {
  %ax = arith.muli %a, %x : index
  %bx = arith.muli %b, %x : index
  %cx = arith.muli %c, %x : index
  %s1 = arith.addi %ax, %bx : index
  %res = arith.addi %s1, %cx : index
  return %res : index
}
