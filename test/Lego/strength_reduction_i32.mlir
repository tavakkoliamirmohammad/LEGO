// RUN: lego-opt %s --lego-strength-reduction | FileCheck %s
// Strength reduction must keep the operand's integer type: an i32 remui/muli/divui
// by a power of two lowers to andi/shli/shrui with an i32 constant, not an index one.

// CHECK-LABEL: func.func @i32_pow2
// CHECK-DAG: %[[S:.*]] = arith.constant 3 : i32
// CHECK-DAG: %[[M:.*]] = arith.constant 15 : i32
// CHECK: arith.andi %{{.*}}, %[[M]] : i32
// CHECK: arith.shli %{{.*}}, %[[S]] : i32
// CHECK: arith.shrui %{{.*}}, %[[S]] : i32
func.func @i32_pow2(%x: i32) -> (i32, i32, i32) {
  %c16 = arith.constant 16 : i32
  %c8 = arith.constant 8 : i32
  %r = arith.remui %x, %c16 : i32
  %m = arith.muli %x, %c8 : i32
  %d = arith.divui %x, %c8 : i32
  return %r, %m, %d : i32, i32, i32
}
