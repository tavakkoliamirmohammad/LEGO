// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// CHECK-LABEL: func.func @noop_passthrough
// CHECK: arith.addi
// CHECK: return
func.func @noop_passthrough(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  return %c : i32
}
