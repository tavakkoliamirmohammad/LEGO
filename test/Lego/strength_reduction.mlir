// RUN: lego-opt %s -lego-strength-reduction | FileCheck %s
//
// Tests for the lego-strength-reduction pass:
//   muli(x, 2^k)  → shli(x, k)
//   divui(x, 2^k) → shrui(x, k)
//   remui(x, 2^k) → andi(x, 2^k - 1)

// CHECK-LABEL: func.func @muli_pow2
// CHECK-SAME:  (%[[X:.*]]: index)
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK:       %[[S1:.*]] = arith.shli %[[X]], %[[C1]] : index
// CHECK:       %[[S2:.*]] = arith.shli %[[X]], %[[C3]] : index
// CHECK:       %[[S3:.*]] = arith.shli %[[X]], %[[C5]] : index
// CHECK:       return %[[S1]], %[[S2]], %[[S3]] : index, index, index
func.func @muli_pow2(%x: index) -> (index, index, index) {
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %a = arith.muli %x, %c2 : index
  %b = arith.muli %x, %c8 : index
  %c = arith.muli %x, %c32 : index
  return %a, %b, %c : index, index, index
}

// CHECK-LABEL: func.func @divui_pow2
// CHECK-SAME:  (%[[X:.*]]: index)
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[S1:.*]] = arith.shrui %[[X]], %[[C2]] : index
// CHECK:       %[[S2:.*]] = arith.shrui %[[X]], %[[C4]] : index
// CHECK:       return %[[S1]], %[[S2]] : index, index
func.func @divui_pow2(%x: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %a = arith.divui %x, %c4 : index
  %b = arith.divui %x, %c16 : index
  return %a, %b : index, index
}

// CHECK-LABEL: func.func @remui_pow2
// CHECK-SAME:  (%[[X:.*]]: index)
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C63:.*]] = arith.constant 63 : index
// CHECK:       %[[S1:.*]] = arith.andi %[[X]], %[[C3]] : index
// CHECK:       %[[S2:.*]] = arith.andi %[[X]], %[[C63]] : index
// CHECK:       return %[[S1]], %[[S2]] : index, index
func.func @remui_pow2(%x: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %a = arith.remui %x, %c4 : index
  %b = arith.remui %x, %c64 : index
  return %a, %b : index, index
}

// Non-power-of-2 constants should be left alone.
// CHECK-LABEL: func.func @non_pow2_unchanged
// CHECK-SAME:  (%[[X:.*]]: index)
// CHECK:       arith.muli %[[X]], %{{.*}} : index
// CHECK:       arith.divui %[[X]], %{{.*}} : index
// CHECK:       arith.remui %[[X]], %{{.*}} : index
func.func @non_pow2_unchanged(%x: index) -> (index, index, index) {
  %c3 = arith.constant 3 : index
  %c10 = arith.constant 10 : index
  %c7 = arith.constant 7 : index
  %a = arith.muli %x, %c3 : index
  %b = arith.divui %x, %c10 : index
  %c = arith.remui %x, %c7 : index
  return %a, %b, %c : index, index, index
}

// Multiply-by-1 is folded by canonicalization (not strength-reduced).
// CHECK-LABEL: func.func @muli_by_one
// CHECK-NOT:   arith.shli
// CHECK:       return %[[X:.*]] : index
func.func @muli_by_one(%x: index) -> index {
  %c1 = arith.constant 1 : index
  %a = arith.muli %x, %c1 : index
  return %a : index
}

// Both operands are power-of-2 constants: should reduce then fold.
// CHECK-LABEL: func.func @both_const_pow2
// CHECK:       %[[C:.*]] = arith.constant 32 : index
// CHECK:       return %[[C]] : index
func.func @both_const_pow2() -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %a = arith.muli %c4, %c8 : index
  return %a : index
}

// divui(x, 1) and remui(x, 1) are canonicalized away (not strength-reduced).
// divui(x,1) → x, remui(x,1) → 0
// CHECK-LABEL: func.func @div_rem_by_one
// CHECK-NOT:   arith.shrui
// CHECK-NOT:   arith.andi
// CHECK:       return
func.func @div_rem_by_one(%x: index) -> (index, index) {
  %c1 = arith.constant 1 : index
  %a = arith.divui %x, %c1 : index
  %b = arith.remui %x, %c1 : index
  return %a, %b : index, index
}

// Commutative muli: constant on left side should also be reduced.
// CHECK-LABEL: func.func @muli_const_lhs
// CHECK-SAME:  (%[[X:.*]]: index)
// CHECK:       %[[C:.*]] = arith.constant 4 : index
// CHECK:       %[[R:.*]] = arith.shli %[[X]], %[[C]] : index
// CHECK:       return %[[R]] : index
func.func @muli_const_lhs(%x: index) -> index {
  %c16 = arith.constant 16 : index
  %a = arith.muli %c16, %x : index
  return %a : index
}
