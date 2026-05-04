// RUN: lego-opt %s --canonicalize --cse --lego-vectorize-filtered-reduce | FileCheck %s

// Filtered (predicated) reduction:
//   acc = identity
//   for i in 0..N:
//     if cond[i] <pred> threshold:
//       acc = combine(acc, A[i])
//
// Should rewrite to a strip-mined vector loop emitting:
//   - vector.transfer_read of A[i:i+L] and cond[i:i+L]
//   - vector.broadcast of threshold
//   - arith.cmpf to build a vector<L x i1> mask
//   - arith.select(mask, v_a, identity) — non-passing lanes get identity
//   - combine_op (arith.addf / mulf / maximumf / minimumf)
//   - vector.reduction at loop exit, combined with original initAcc
//   - tail loop preserving the original scalar body for (N mod L)

// CHECK-LABEL: func.func @filt_sum_kernel
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK-SAME: vector<{{[0-9]+}}xf32>
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: arith.cmpf
// CHECK: arith.select
// CHECK: arith.addf
// CHECK: vector.reduction <add>
func.func @filt_sum_kernel(%A: memref<1024xf32>, %cond: memref<1024xf32>) -> f32 {
  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %z     = arith.constant 0.000000e+00 : f32
  %r = scf.for %i = %c0 to %c1024 step %c1
               iter_args(%acc = %z) -> f32 {
    %v = memref.load %A[%i]    : memref<1024xf32>
    %c = memref.load %cond[%i] : memref<1024xf32>
    %p = arith.cmpf ogt, %c, %z : f32
    %new = scf.if %p -> f32 {
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    } else {
      scf.yield %acc : f32
    }
    scf.yield %new : f32
  }
  return %r : f32
}
