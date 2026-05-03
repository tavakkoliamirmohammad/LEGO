// RUN: lego-opt %s --lego-to-arm-neon | FileCheck %s

// End-to-end test for the --lego-to-arm-neon pipeline.
// Checks that a SAXPY kernel (y[i] = a*x[i] + y[i]) fully lowers to LLVM
// dialect. The ARM NEON pipeline is structurally identical to lego-to-x86-vector
// at v1: lego-vectorize emits vector<8xf64> ops (AVX-512 default width); the
// LLVM AArch64 backend will split to NEON-width (2xf64) pairs when targeting
// an aarch64 triple. Proper NEON-width vector selection (2 f64 / 4 f32 lanes)
// is deferred to R15.
//
// CHECK-LABEL: llvm.func @saxpy_neon
// CHECK-NOT: lego.
// CHECK-NOT: arith.
// CHECK-NOT: scf.for
// CHECK-NOT: vector.
// CHECK: vector<8xf64>
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
