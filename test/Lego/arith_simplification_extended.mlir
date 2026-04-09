// RUN: lego-opt %s -lego-arith-simplification | FileCheck %s

// ============================================================================
// Extended tests for lego-arith-simplification patterns not covered by
// simplifier_coverage.mlir or distributive_factor.mlir.
//
// Patterns tested:
//   - SimplifyRemOfRem:     (x % d) % d  ->  x % d
//   - ExtendedSimplifyDivId: (q*s + r) / (k*s) -> q/k + ((q%k)*s + r)/(k*s)
//   - ExtendedSimplifyRemId: (q*s + r) % (k*s) -> ((q%k)*s + r) % (k*s)
//   - SimplifyMixedRadixDiv: divui(remui(a,n)*m + remui(b,m), n*m) -> 0
//   - SimplifyMixedRadixRem: remui(remui(a,n)*m + remui(b,m), n*m) -> identity
// ============================================================================

// --- SimplifyRemOfRem: (x % d) % d -> x % d ---
// CHECK-LABEL: func.func @rem_of_rem
// CHECK-SAME:  (%[[X:.*]]: index, %[[D:.*]]: index)
// CHECK:       %[[R:.*]] = arith.remui %[[X]], %[[D]] : index
// CHECK-NOT:   arith.remui %[[R]], %[[D]]
// CHECK:       return %[[R]] : index
func.func @rem_of_rem(%x: index, %d: index) -> index {
  %r1 = arith.remui %x, %d : index
  %r2 = arith.remui %r1, %d : index
  return %r2 : index
}

// --- SimplifyRemOfRem: triple application ---
// CHECK-LABEL: func.func @rem_of_rem_triple
// CHECK-SAME:  (%[[X:.*]]: index, %[[D:.*]]: index)
// CHECK:       %[[R:.*]] = arith.remui %[[X]], %[[D]] : index
// CHECK-NOT:   arith.remui %[[R]], %[[D]]
// CHECK:       return %[[R]] : index
func.func @rem_of_rem_triple(%x: index, %d: index) -> index {
  %r1 = arith.remui %x, %d : index
  %r2 = arith.remui %r1, %d : index
  %r3 = arith.remui %r2, %d : index
  return %r3 : index
}

// --- SimplifyRemOfRem: different divisors should not simplify ---
// CHECK-LABEL: func.func @rem_different_divisors
// CHECK-SAME:  (%[[X:.*]]: index, %[[D1:.*]]: index, %[[D2:.*]]: index)
// CHECK:       arith.remui %[[X]], %[[D1]] : index
// CHECK:       arith.remui %{{.*}}, %[[D2]] : index
func.func @rem_different_divisors(%x: index, %d1: index, %d2: index) -> index {
  %r1 = arith.remui %x, %d1 : index
  %r2 = arith.remui %r1, %d2 : index
  return %r2 : index
}

// --- ExtendedSimplifyDivId: (q*s + r) / (k*s) -> q/k + ((q%k)*s + r)/(k*s) ---
// When the numerator contains a shared factor with the divisor.
// CHECK-LABEL: func.func @extended_div_shared_factor
// CHECK-SAME:  (%[[Q:.*]]: index, %[[S:.*]]: index, %[[K:.*]]: index, %[[R:.*]]: index)
// CHECK:       %[[QDIVK:.*]] = arith.divui %[[Q]], %[[K]] : index
// CHECK:       %[[QREMK:.*]] = arith.remui %[[Q]], %[[K]] : index
// CHECK:       arith.muli %[[QREMK]], %[[S]] : index
// CHECK:       arith.addi
// CHECK:       arith.divui
// CHECK:       arith.addi %[[QDIVK]],
func.func @extended_div_shared_factor(%q: index, %s: index, %k: index, %r: index) -> index {
  %qs = arith.muli %q, %s : index
  %num = arith.addi %qs, %r : index
  %ks = arith.muli %k, %s : index
  %res = arith.divui %num, %ks : index
  return %res : index
}

// --- ExtendedSimplifyRemId: (q*s + r) % (k*s) -> ((q%k)*s + r) % (k*s) ---
// CHECK-LABEL: func.func @extended_rem_shared_factor
// CHECK-SAME:  (%[[Q:.*]]: index, %[[S:.*]]: index, %[[K:.*]]: index, %[[R:.*]]: index)
// CHECK:       %[[QREMK:.*]] = arith.remui %[[Q]], %[[K]] : index
// CHECK:       arith.muli %[[QREMK]], %[[S]] : index
// CHECK:       arith.addi
// CHECK:       arith.remui
func.func @extended_rem_shared_factor(%q: index, %s: index, %k: index, %r: index) -> index {
  %qs = arith.muli %q, %s : index
  %num = arith.addi %qs, %r : index
  %ks = arith.muli %k, %s : index
  %res = arith.remui %num, %ks : index
  return %res : index
}

// --- SimplifyMixedRadixDiv: divui(remui(a,n)*m + remui(b,m), n*m) -> 0 ---
// CHECK-LABEL: func.func @mixed_radix_div_zero
// CHECK-SAME:  (%[[A:.*]]: index, %[[B:.*]]: index, %[[N:.*]]: index, %[[M:.*]]: index)
// CHECK:       %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:       return %[[ZERO]] : index
func.func @mixed_radix_div_zero(%a: index, %b: index, %n: index, %m: index) -> index {
  %an = arith.remui %a, %n : index
  %bm = arith.remui %b, %m : index
  %hi = arith.muli %an, %m : index
  %sum = arith.addi %hi, %bm : index
  %nm = arith.muli %n, %m : index
  %res = arith.divui %sum, %nm : index
  return %res : index
}

// --- SimplifyMixedRadixRem: remui(remui(a,n)*m + remui(b,m), n*m) -> identity ---
// CHECK-LABEL: func.func @mixed_radix_rem_identity
// CHECK-SAME:  (%[[A:.*]]: index, %[[B:.*]]: index, %[[N:.*]]: index, %[[M:.*]]: index)
// CHECK:       %[[AN:.*]] = arith.remui %[[A]], %[[N]] : index
// CHECK:       %[[BM:.*]] = arith.remui %[[B]], %[[M]] : index
// CHECK:       %[[HI:.*]] = arith.muli %[[AN]], %[[M]] : index
// CHECK:       %[[SUM:.*]] = arith.addi %[[HI]], %[[BM]] : index
// CHECK-NOT:   arith.remui %[[SUM]]
// CHECK:       return %[[SUM]] : index
func.func @mixed_radix_rem_identity(%a: index, %b: index, %n: index, %m: index) -> index {
  %an = arith.remui %a, %n : index
  %bm = arith.remui %b, %m : index
  %hi = arith.muli %an, %m : index
  %sum = arith.addi %hi, %bm : index
  %nm = arith.muli %n, %m : index
  %res = arith.remui %sum, %nm : index
  return %res : index
}

// --- SimplifyMixedRadixDiv with swapped divisor operands ---
// divisor = m*n instead of n*m: should still match due to commutativity.
// CHECK-LABEL: func.func @mixed_radix_div_commuted_divisor
// CHECK-SAME:  (%[[A:.*]]: index, %[[B:.*]]: index, %[[N:.*]]: index, %[[M:.*]]: index)
// CHECK:       %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:       return %[[ZERO]] : index
func.func @mixed_radix_div_commuted_divisor(%a: index, %b: index, %n: index, %m: index) -> index {
  %an = arith.remui %a, %n : index
  %bm = arith.remui %b, %m : index
  %hi = arith.muli %an, %m : index
  %sum = arith.addi %hi, %bm : index
  %mn = arith.muli %m, %n : index
  %res = arith.divui %sum, %mn : index
  return %res : index
}

// --- SimplifyDivConst: (x + 20) / 10 -> x/10 + 2 ---
// CHECK-LABEL: func.func @div_const_multiple
// CHECK-SAME:  (%[[X:.*]]: index)
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[D:.*]] = arith.divui %[[X]], %{{.*}} : index
// CHECK:       %[[R:.*]] = arith.addi %[[D]], %[[C2]] : index
// CHECK:       return %[[R]] : index
func.func @div_const_multiple(%x: index) -> index {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %sum = arith.addi %x, %c20 : index
  %res = arith.divui %sum, %c10 : index
  return %res : index
}

// --- SimplifyDivConst: constant not divisible should NOT fire ---
// CHECK-LABEL: func.func @div_const_not_divisible
// CHECK:       arith.addi
// CHECK:       arith.divui
func.func @div_const_not_divisible(%x: index) -> index {
  %c10 = arith.constant 10 : index
  %c15 = arith.constant 15 : index
  %sum = arith.addi %x, %c15 : index
  %res = arith.divui %sum, %c10 : index
  return %res : index
}

// --- SimplifyDivOfRem: (x % d) / d -> 0 ---
// CHECK-LABEL: func.func @div_of_rem_zero
// CHECK-SAME:  (%[[X:.*]]: index, %[[D:.*]]: index)
// CHECK:       %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:       return %[[ZERO]] : index
func.func @div_of_rem_zero(%x: index, %d: index) -> index {
  %r = arith.remui %x, %d : index
  %res = arith.divui %r, %d : index
  return %res : index
}

// --- ReconstructId: (x / d) * d + (x % d) -> x ---
// CHECK-LABEL: func.func @reconstruct_identity
// CHECK-SAME:  (%[[X:.*]]: index, %[[D:.*]]: index)
// CHECK:       return %[[X]] : index
func.func @reconstruct_identity(%x: index, %d: index) -> index {
  %div = arith.divui %x, %d : index
  %rem = arith.remui %x, %d : index
  %mul = arith.muli %div, %d : index
  %res = arith.addi %mul, %rem : index
  return %res : index
}

// --- ReconstructId: commuted addend order ---
// (x % d) + (x / d) * d -> x
// CHECK-LABEL: func.func @reconstruct_identity_commuted
// CHECK-SAME:  (%[[X:.*]]: index, %[[D:.*]]: index)
// CHECK:       return %[[X]] : index
func.func @reconstruct_identity_commuted(%x: index, %d: index) -> index {
  %div = arith.divui %x, %d : index
  %rem = arith.remui %x, %d : index
  %mul = arith.muli %div, %d : index
  %res = arith.addi %rem, %mul : index
  return %res : index
}
