// RUN: lego-opt %s --canonicalize --cse --lego-vectorize-scan | FileCheck %s

// Inclusive prefix-scan / cumulative-sum loop:
//   acc = 0
//   for i in 0..N:
//     acc = acc + A[i]
//     B[i] = acc
//
// Should rewrite to a strip-mined vector loop emitting:
//   - vector.transfer_read of A[i:i+L]
//   - log2(L) Hillis-Steele stages: vector.shuffle + arith.addf
//   - arith.addf with broadcast carry
//   - vector.transfer_write of B[i:i+L]
//   - vector.extract of the last lane → broadcast → next iter carry
//   - tail loop preserves the original scalar body for (N mod L)

// CHECK-LABEL: func.func @scan_kernel
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK-SAME: vector<{{[0-9]+}}xf32>
// CHECK: vector.transfer_read
// CHECK: vector.shuffle
// CHECK: arith.addf
// CHECK: vector.shuffle
// CHECK: arith.addf
// CHECK: vector.transfer_write
// CHECK: vector.extract
// CHECK: vector.broadcast
func.func @scan_kernel(%A: memref<1024xf32>, %B: memref<1024xf32>) {
  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %z     = arith.constant 0.000000e+00 : f32
  %r = scf.for %i = %c0 to %c1024 step %c1
               iter_args(%acc = %z) -> f32 {
    %v = memref.load %A[%i] : memref<1024xf32>
    %s = arith.addf %acc, %v : f32
    memref.store %s, %B[%i] : memref<1024xf32>
    scf.yield %s : f32
  }
  return
}
