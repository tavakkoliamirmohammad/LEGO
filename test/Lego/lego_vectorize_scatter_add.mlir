// RUN: lego-opt %s --lego-vectorize-scatter-add | FileCheck %s

// Scatter-add / histogram loop:
//   for i in 0..N:
//     b = bin[i]
//     count[b] = count[b] + A[i]
//
// Should rewrite to a strip-mined vector loop emitting:
//   - vector.transfer_read of bin[i:i+L]
//   - L-1 vector.shuffle + arith.cmpi + arith.ori + vector.reduction <or>
//     for cross-lane conflict detection
//   - scf.if (any conflict) { scalar fallback } else { gather + addf + scatter }
//   - tail loop preserving the original scalar body for (N mod L) iterations

// CHECK-LABEL: func.func @scatter_add_kernel
// CHECK: vector.transfer_read
// CHECK: vector.shuffle
// CHECK: arith.cmpi eq
// CHECK: vector.reduction <or>
// CHECK: scf.if
// CHECK: vector.gather
// CHECK: arith.addf
// CHECK: vector.scatter
func.func @scatter_add_kernel(%bin:   memref<1024xi32>,
                              %A:     memref<1024xf32>,
                              %count: memref<256xf32>) {
  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  scf.for %i = %c0 to %c1024 step %c1 {
    %b32 = memref.load %bin[%i] : memref<1024xi32>
    %b   = arith.index_cast %b32 : i32 to index
    %v   = memref.load %count[%b] : memref<256xf32>
    %d   = memref.load %A[%i] : memref<1024xf32>
    %s   = arith.addf %v, %d : f32
    memref.store %s, %count[%b] : memref<256xf32>
  }
  return
}
