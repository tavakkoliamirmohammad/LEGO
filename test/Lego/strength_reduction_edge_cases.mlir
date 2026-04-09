// RUN: lego-opt %s -lego-strength-reduction | FileCheck %s
//
// Edge-case tests for the lego-strength-reduction pass.
// Complements strength_reduction.mlir with boundary and unusual inputs.

// --- Large power of 2: muli(x, 2^10=1024) -> shli(x, 10) ---
// CHECK-LABEL: func.func @muli_large_pow2
// CHECK-SAME:  (%[[X:.*]]: index)
// CHECK:       %[[C10:.*]] = arith.constant 10 : index
// CHECK:       %[[R:.*]] = arith.shli %[[X]], %[[C10]] : index
// CHECK:       return %[[R]] : index
func.func @muli_large_pow2(%x: index) -> index {
  %c1024 = arith.constant 1024 : index
  %a = arith.muli %x, %c1024 : index
  return %a : index
}

// --- divui(x, 2^10=1024) -> shrui(x, 10) ---
// CHECK-LABEL: func.func @divui_large_pow2
// CHECK-SAME:  (%[[X:.*]]: index)
// CHECK:       %[[C10:.*]] = arith.constant 10 : index
// CHECK:       %[[R:.*]] = arith.shrui %[[X]], %[[C10]] : index
// CHECK:       return %[[R]] : index
func.func @divui_large_pow2(%x: index) -> index {
  %c1024 = arith.constant 1024 : index
  %a = arith.divui %x, %c1024 : index
  return %a : index
}

// --- remui(x, 2^10=1024) -> andi(x, 1023) ---
// CHECK-LABEL: func.func @remui_large_pow2
// CHECK-SAME:  (%[[X:.*]]: index)
// CHECK:       %[[C1023:.*]] = arith.constant 1023 : index
// CHECK:       %[[R:.*]] = arith.andi %[[X]], %[[C1023]] : index
// CHECK:       return %[[R]] : index
func.func @remui_large_pow2(%x: index) -> index {
  %c1024 = arith.constant 1024 : index
  %a = arith.remui %x, %c1024 : index
  return %a : index
}

// --- Power of 2 = 2 (smallest non-trivial) ---
// CHECK-LABEL: func.func @muli_pow2_smallest
// CHECK-SAME:  (%[[X:.*]]: index)
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[R:.*]] = arith.shli %[[X]], %[[C1]] : index
// CHECK:       return %[[R]] : index
func.func @muli_pow2_smallest(%x: index) -> index {
  %c2 = arith.constant 2 : index
  %a = arith.muli %x, %c2 : index
  return %a : index
}

// --- Multiply by 0: constant-folded to 0 (not strength-reduced) ---
// CHECK-LABEL: func.func @muli_by_zero
// CHECK-NOT:   arith.shli
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       return %[[C0]] : index
func.func @muli_by_zero(%x: index) -> index {
  %c0 = arith.constant 0 : index
  %a = arith.muli %x, %c0 : index
  return %a : index
}

// --- muli by 1: constant-folded to identity (not strength-reduced) ---
// CHECK-LABEL: func.func @muli_by_one_skip
// CHECK-NOT:   arith.shli
// CHECK:       return %{{.*}} : index
func.func @muli_by_one_skip(%x: index) -> index {
  %c1 = arith.constant 1 : index
  %a = arith.muli %x, %c1 : index
  return %a : index
}

// --- divui by 1: constant-folded to identity (not strength-reduced) ---
// CHECK-LABEL: func.func @divui_by_one_skip
// CHECK-NOT:   arith.shrui
// CHECK:       return %{{.*}} : index
func.func @divui_by_one_skip(%x: index) -> index {
  %c1 = arith.constant 1 : index
  %a = arith.divui %x, %c1 : index
  return %a : index
}

// --- remui by 1: constant-folded to 0 (not strength-reduced) ---
// CHECK-LABEL: func.func @remui_by_one_skip
// CHECK-NOT:   arith.andi
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       return %[[C0]] : index
func.func @remui_by_one_skip(%x: index) -> index {
  %c1 = arith.constant 1 : index
  %a = arith.remui %x, %c1 : index
  return %a : index
}

// --- Non-constant divisor: should not be strength-reduced ---
// CHECK-LABEL: func.func @non_constant_divisor
// CHECK-SAME:  (%[[X:.*]]: index, %[[D:.*]]: index)
// CHECK:       arith.muli %[[X]], %[[D]]
// CHECK:       arith.divui %[[X]], %[[D]]
// CHECK:       arith.remui %[[X]], %[[D]]
func.func @non_constant_divisor(%x: index, %d: index) -> (index, index, index) {
  %a = arith.muli %x, %d : index
  %b = arith.divui %x, %d : index
  %c = arith.remui %x, %d : index
  return %a, %b, %c : index, index, index
}

// --- Multiple operations in chain ---
// muli(divui(x, 4), 8) -> shli(shrui(x, 2), 3)
// CHECK-LABEL: func.func @chain_div_then_mul
// CHECK-SAME:  (%[[X:.*]]: index)
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK:       %[[SHR:.*]] = arith.shrui %[[X]], %[[C2]] : index
// CHECK:       %[[SHL:.*]] = arith.shli %[[SHR]], %[[C3]] : index
// CHECK:       return %[[SHL]] : index
func.func @chain_div_then_mul(%x: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %div = arith.divui %x, %c4 : index
  %mul = arith.muli %div, %c8 : index
  return %mul : index
}

// --- 2^16 = 65536 ---
// CHECK-LABEL: func.func @pow2_16
// CHECK-SAME:  (%[[X:.*]]: index)
// CHECK:       %[[C16:.*]] = arith.constant 16 : index
// CHECK:       %[[R:.*]] = arith.shli %[[X]], %[[C16]] : index
// CHECK:       return %[[R]] : index
func.func @pow2_16(%x: index) -> index {
  %c65536 = arith.constant 65536 : index
  %a = arith.muli %x, %c65536 : index
  return %a : index
}
