// RUN: lego-opt %s --lego-vectorize-compact | FileCheck %s

// Compaction loop:
//   k = 0
//   for i in 0..N:
//     if cond[i] > 0.0:
//       B[k] = A[i]
//       k = k + 1
//
// Should rewrite to a strip-mined vector loop emitting:
//   - vector.transfer_read for both A[i] and cond[i]
//   - arith.cmpf to build a vector<L x i1> mask
//   - vector.compressstore — the *non-clang* op (vpcompressps on x86)
//   - math.ctpop on the bitcast mask to advance the iter_arg
//   - tail loop preserving the original scalar body for (N mod L)

// CHECK-LABEL: func.func @compact_kernel
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: arith.cmpf
// CHECK: vector.compressstore
// CHECK: math.ctpop
// CHECK-NOT: arith.cmpf {{.*}} : f32   // scalar cmp should be in tail only
func.func @compact_kernel(%A: memref<1024xf32>, %cond: memref<1024xf32>,
                          %B: memref<1024xf32>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %fzero = arith.constant 0.0 : f32
  %k_final = scf.for %i = %c0 to %c1024 step %c1
                     iter_args(%k = %c0) -> (index) {
    %v = memref.load %A[%i] : memref<1024xf32>
    %c = memref.load %cond[%i] : memref<1024xf32>
    %p = arith.cmpf ogt, %c, %fzero : f32
    %k_new = scf.if %p -> index {
      memref.store %v, %B[%k] : memref<1024xf32>
      %k1 = arith.addi %k, %c1 : index
      scf.yield %k1 : index
    } else {
      scf.yield %k : index
    }
    scf.yield %k_new : index
  }
  return %k_final : index
}
