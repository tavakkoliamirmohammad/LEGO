// RUN: lego-opt %s --lego-to-x86-vector | FileCheck %s

// End-to-end test for the --lego-to-x86-vector pipeline.
// Checks that a SAXPY kernel (y[i] = a*x[i] + y[i]) fully lowers to LLVM
// dialect with vector<8xf64> operations (AVX-512 lane width for f64).
//
// The pipeline: buildLegoLowerPipeline → lego-vectorize → convert-vector-to-llvm
// → SCF→CF → Arith/MemRef/Func/CF → LLVM.  No lego/arith/scf/vector ops
// should survive.

// CHECK-LABEL: llvm.func @saxpy
// CHECK-NOT: lego.
// CHECK-NOT: arith.
// CHECK-NOT: scf.for
// CHECK-NOT: vector.
// CHECK: vector<8xf64>
// CHECK: llvm.fmul
// CHECK: llvm.fadd
// CHECK: llvm.return
func.func @saxpy(%a: f64, %X: memref<?xf64>, %Y: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %xi = memref.load %X[%i] : memref<?xf64>
    %yi = memref.load %Y[%i] : memref<?xf64>
    %p  = arith.mulf %a, %xi : f64
    %s  = arith.addf %p, %yi : f64
    memref.store %s, %Y[%i] : memref<?xf64>
  }
  return
}
