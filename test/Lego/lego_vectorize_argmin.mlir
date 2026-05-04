// RUN: lego-opt %s --canonicalize --cse --lego-vectorize-argmin | FileCheck %s

// Argmin loop:
//   m = +inf, mi = 0
//   for i in 0..N:
//     if A[i] < m:
//       m = A[i]
//       mi = i
//
// Should rewrite to a strip-mined vector loop with vector accumulators
// and a final scalar reduction:
//   - vector.transfer_read of A[i:i+L]
//   - vector.step + vector.broadcast(i) for per-lane indices
//   - arith.cmpf olt + arith.minimumf + arith.select for paired update
//   - vector.reduction <minimumf> for scalar min
//   - cmpf-oeq + select + vector.reduction <minui> for scalar idx
//   - tail loop preserves the original scalar body for (N mod L)

// CHECK-LABEL: func.func @argmin_kernel
// CHECK: vector.step
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK-SAME: vector<{{[0-9]+}}xf32>
// CHECK-SAME: vector<{{[0-9]+}}xindex>
// CHECK: vector.transfer_read
// CHECK: arith.cmpf olt
// CHECK: arith.minimumf
// CHECK: arith.select
// CHECK: vector.reduction <minimumf>
// CHECK: vector.reduction <minui>
func.func @argmin_kernel(%A: memref<1024xf32>) -> (f32, index) {
  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %inf   = arith.constant 1.000000e+30 : f32
  %r:2 = scf.for %i = %c0 to %c1024 step %c1
                 iter_args(%m = %inf, %mi = %c0) -> (f32, index) {
    %v = memref.load %A[%i] : memref<1024xf32>
    %p = arith.cmpf olt, %v, %m : f32
    %new_m = scf.if %p -> (f32) {
      %v2 = memref.load %A[%i] : memref<1024xf32>
      scf.yield %v2 : f32
    } else {
      scf.yield %m : f32
    }
    %new_mi = arith.select %p, %i, %mi : index
    scf.yield %new_m, %new_mi : f32, index
  }
  return %r#0, %r#1 : f32, index
}
