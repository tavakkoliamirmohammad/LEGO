// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// Strided access pattern tests.
// TierA classifies: coeff > 1 with constant factor → Strided.
// TierB classifies: uniform non-unit differences → Strided.
//
// R20 deinterleave: for constant element-strides in {2, 4, 8} with
// stride * Ln <= 256 elements, the vectorizer emits S transfer_reads +
// vector.shuffle chains instead of vector.gather. This mirrors what
// clang/gcc produce for "load + vpermt2ps" deinterleave sequences.
// Large strides (> 8) or runtime strides fall back to vector.gather.

// ---------------------------------------------------------------------------
// Test 1: Column-major access — stride 64 elements (> 8) → gather.
// addr(i) = i * 64: large constant stride → falls back to vector.gather.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @col_major_static_stride
// CHECK: vector.gather
// CHECK-NOT: vector.shuffle
func.func @col_major_static_stride(%A: memref<1024xf64>, %B: memref<1024xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %off = arith.muli %i, %c64 : index
    %v = memref.load %A[%off] : memref<1024xf64>
    memref.store %v, %B[%i] : memref<1024xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 2: Runtime stride (non-constant) → NonAffine → vector.gather.
// TierA yields NonAffine (muli by non-constant). Falls back to gather.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @col_major_runtime_stride
// CHECK: vector.gather
func.func @col_major_runtime_stride(%A: memref<?xf64>, %B: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %off = arith.muli %i, %N : index
    %v = memref.load %A[%off] : memref<?xf64>
    memref.store %v, %B[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 3: Stride 8 elements (f64, 8*8=64 bytes) — in {2,4,8}, stride*Ln=64.
// R20 deinterleave fires: one wide transfer_read + log2(stride) deinterleaves,
// no gather.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @stride_equals_regwidth
// CHECK: vector.transfer_read
// CHECK: vector.deinterleave
// CHECK-NOT: vector.gather
func.func @stride_equals_regwidth(%A: memref<?xf64>, %B: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %off = arith.muli %i, %c8 : index
    %v = memref.load %A[%off] : memref<?xf64>
    memref.store %v, %B[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 4: Two strided accesses with different strides.
// A[i*4]: f64, element-stride=4 (power-of-2, ≤ 8) → R20 deinterleave path.
// B[i*16]: f64, element-stride=16 (non-pow-2-but-≤16, stride·Ln=128 ≤ 256)
//          → wide-load + vector.shuffle path (since 16 isn't in the
//          deinterleave fast set {2,4,8}).  Was vector.gather before the
//          shuffle generalisation; the change wins on stride-7-class
//          kernels where gather was microcoded on Zen 4.
// C[i]: unit-stride store → transfer_write.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @two_strided_accesses
// CHECK: vector.transfer_read
// CHECK: vector.deinterleave
// CHECK: vector.shuffle
// CHECK-NOT: vector.gather
// CHECK: vector.transfer_write
func.func @two_strided_accesses(%A: memref<?xf64>, %B: memref<?xf64>, %C: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %off_a = arith.muli %i, %c4 : index
    %va = memref.load %A[%off_a] : memref<?xf64>
    %off_b = arith.muli %i, %c16 : index
    %vb = memref.load %B[%off_b] : memref<?xf64>
    %s = arith.addf %va, %vb : f64
    memref.store %s, %C[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 5: Mixed unit + strided in same loop.
// A[i] is unit-stride → transfer_read of vector<8xf64>.
// B[i*4]: f64, element-stride=4 (in {2,4,8}) → R20 deinterleave path: one
// wide read of vector<32xf64> followed by log2(4)=2 ``vector.deinterleave``
// ops to peel out every 4th element.
// The result: transfer_reads + deinterleave(s) + transfer_write. No gather.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @mixed_unit_strided
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<32xf64>
// CHECK: vector.deinterleave
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
// CHECK-NOT: vector.gather
func.func @mixed_unit_strided(%A: memref<?xf64>, %B: memref<?xf64>, %C: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %va = memref.load %A[%i] : memref<?xf64>
    %off_b = arith.muli %i, %c4 : index
    %vb = memref.load %B[%off_b] : memref<?xf64>
    %s = arith.addf %va, %vb : f64
    memref.store %s, %C[%i] : memref<?xf64>
  }
  return
}
