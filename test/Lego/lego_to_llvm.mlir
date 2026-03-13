// RUN: lego-opt --lego-to-llvm %s | FileCheck %s

// Test that the lego-to-llvm pipeline fully lowers a simple layout to LLVM dialect.

// CHECK-NOT: lego.
// CHECK-NOT: arith.
// CHECK-NOT: scf.for
// CHECK: llvm.func

func.func @test_row_layout_apply(%arg0: index, %arg1: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %layout = lego.row [%c4, %c8] : !lego.layout
  %flat = lego.apply %layout(%arg0, %arg1) : !lego.layout
  return %flat : index
}

// -----

// CHECK-NOT: lego.
// CHECK: llvm.func

func.func @test_regp_apply(%arg0: index, %arg1: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %layout = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  %flat = lego.apply %layout(%arg0, %arg1) : !lego.layout
  return %flat : index
}

// -----

// CHECK-NOT: lego.
// CHECK: llvm.func

func.func @test_apply_inverse(%arg0: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %layout = lego.row [%c4, %c8] : !lego.layout
  %i, %j = lego.apply_inverse %layout(%arg0) : !lego.layout -> index, index
  return %i, %j : index, index
}

// -----

// Test a memref load/store through layout transformation lowers to LLVM.
// CHECK-NOT: lego.
// CHECK: llvm.func

func.func @test_transform(%src: memref<32xf32>, %dst: memref<32xf32>, %n: index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %layout = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %i0, %i1 = lego.apply_inverse %layout(%i) : !lego.layout -> index, index
    %new_idx = lego.apply %layout(%i0, %i1) : !lego.layout
    %val = memref.load %src[%i] : memref<32xf32>
    memref.store %val, %dst[%new_idx] : memref<32xf32>
  }
  return
}
