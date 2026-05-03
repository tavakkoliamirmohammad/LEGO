// RUN: lego-opt %s --lego-to-arm-neon | FileCheck %s

// End-to-end test for the --lego-to-arm-neon pipeline.
// R15 complete: lego-vectorize now emits NEON-width vectors when the pipeline
// passes target="neon". For f64: 16-byte NEON register / 8 bytes = 2 lanes.
// So the pipeline emits vector<2xf64> (not the old AVX-512 default vector<8xf64>).
//
// CHECK-LABEL: llvm.func @saxpy_neon
// CHECK-NOT: lego.
// CHECK-NOT: arith.
// CHECK-NOT: scf.for
// CHECK-NOT: vector.
// CHECK: vector<2xf64>
// CHECK: llvm.fmul
// CHECK: llvm.fadd
// CHECK: llvm.return
func.func @saxpy_neon(%a: f64, %X: memref<?xf64>, %Y: memref<?xf64>, %N: index) {
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
