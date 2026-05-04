// RUN: lego-opt %s --canonicalize --cse --lego-vectorize-rle | FileCheck %s

// RLE / edge-detect compaction:
//   k = 0; prev = sentinel
//   for i in 0..N: if A[i] != prev: out[k] = A[i]; prev = A[i]; k++
//
// Should rewrite to a strip-mined vector loop emitting:
//   - vector.transfer_read of A[i:i+L]
//   - vector.shuffle (prev_vec, v) [L-1, L+0, ..., L+(L-2)]
//   - arith.cmpf one for the lane mask
//   - vector.compressstore out[k], mask, v
//   - math.ctpop on bitcast mask to advance k
//   - tail loop preserves the original scalar body for (N mod L)

// CHECK-LABEL: func.func @rle_kernel
// CHECK: scf.for
// CHECK-SAME: iter_args
// CHECK-SAME: vector<{{[0-9]+}}xf32>
// CHECK: vector.transfer_read
// CHECK: vector.shuffle
// CHECK: arith.cmpf one
// CHECK: vector.compressstore
// CHECK: math.ctpop
func.func @rle_kernel(%A: memref<1024xf32>, %out: memref<1024xf32>) -> (index, f32) {
  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %sentinel = arith.constant -1.000000e+30 : f32
  %r:2 = scf.for %i = %c0 to %c1024 step %c1
                 iter_args(%k = %c0, %prev = %sentinel) -> (index, f32) {
    %v = memref.load %A[%i] : memref<1024xf32>
    %p = arith.cmpf one, %v, %prev : f32
    %new:2 = scf.if %p -> (index, f32) {
      memref.store %v, %out[%k] : memref<1024xf32>
      %k1 = arith.addi %k, %c1 : index
      scf.yield %k1, %v : index, f32
    } else {
      scf.yield %k, %prev : index, f32
    }
    scf.yield %new#0, %new#1 : index, f32
  }
  return %r#0, %r#1 : index, f32
}
