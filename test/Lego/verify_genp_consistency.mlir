// RUN: lego-opt -lego-verify-genp-consistency %s 2>&1 | FileCheck %s

// CHECK: error: Inconsistent GenP: apply and inv regions are not bijections.
func.func @inconsistent_genp(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %layout = lego.gen_p [%c4, %c4] apply (%a: index, %b: index) {
    %sum = arith.addi %a, %b : index
    lego.yield %sum : index
  } inv (%flat: index) {
    %z = arith.constant 0 : index
    lego.yield %z, %z : index, index
  } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-NOT: error
func.func @consistent_genp(%i: index, %j: index) -> (index, index) {
  %cc4 = arith.constant 4 : index
  %cc8 = arith.constant 8 : index
  %layout = lego.gen_p [%cc4, %cc8] apply (%a: index, %b: index) {
    %c8 = arith.constant 8 : index
    %t = arith.muli %a, %c8 : index
    %f = arith.addi %t, %b : index
    lego.yield %f : index
  } inv (%flat: index) {
    %c8 = arith.constant 8 : index
    %ii = arith.divui %flat, %c8 : index
    %jj = arith.remui %flat, %c8 : index
    lego.yield %ii, %jj : index, index
  } : !lego.layout
  %flat = lego.apply %layout(%i, %j) : !lego.layout
  %ri, %rj = lego.apply_inverse %layout(%flat) : !lego.layout -> index, index
  return %ri, %rj : index, index
}
