// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// Non-affine and gather pattern tests.
// These are cases where TierA returns NonAffine and TierB probes concrete
// addresses 0..L-1 and detects non-uniform strides → emits vector.gather.

// ---------------------------------------------------------------------------
// Test 1: Morton-style bit-interleave (already covered in lego_vectorize.mlir,
// repeated here as part of the gather matrix for completeness).
// TierA: andi / shli / ori are not handled → NonAffine.
// TierB: probes i=0..7 for the inner j loop, computes addresses, finds
//        non-uniform differences → NonAffine → vector.gather.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @morton_style
// CHECK: vector.gather
func.func @morton_style(%A: memref<?xf64>, %B: memref<?xf64>, %i_outer: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c5 = arith.constant 5 : index
  scf.for %j = %c0 to %c8 step %c1 {
    %ti = arith.andi %i_outer, %c5 : index
    %tj = arith.andi %j, %c5 : index
    %tj_shl = arith.shli %tj, %c1 : index
    %morton = arith.ori %ti, %tj_shl : index
    %v = memref.load %A[%morton] : memref<?xf64>
    memref.store %v, %B[%j] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 2: Indirect indexing: B[idx[i]] — runtime address from another array.
// The outer load of idx is unit-stride; the inner A[raw_idx] is NonAffine
// (TierB: probes concrete idx values → unpredictable → NonAffine → gather).
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @indirect_gather
// CHECK: vector.transfer_read {{.*}} : memref<?xindex>, vector<8xindex>
// CHECK: vector.gather {{.*}} : memref<?xf64>, vector<8xindex>, vector<8xi1>, vector<8xf64>
func.func @indirect_gather(%A: memref<?xf64>, %idx: memref<?xindex>, %B: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %raw_idx = memref.load %idx[%i] : memref<?xindex>
    %v = memref.load %A[%raw_idx] : memref<?xf64>
    memref.store %v, %B[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 3: pow2_mod: addr = i & 31.
// For i=0..7: 0&31=0, 1&31=1, ..., 7&31=7 → uniform unit stride.
// TierB detects firstStep=1 uniform → classified as Unit → transfer_read,
// NOT gather. This is correct: the masking with 31 is a no-op for i < 32.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @pow2_mod
// CHECK: vector.transfer_read
// CHECK-NOT: vector.gather
func.func @pow2_mod(%A: memref<?xf64>, %B: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c31 = arith.constant 31 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %masked = arith.andi %i, %c31 : index
    %v = memref.load %A[%masked] : memref<?xf64>
    memref.store %v, %B[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 4: XOR pattern: addr = (i << 1) ^ 3.
// For i=0..7: 0^3=3, 2^3=1, 4^3=7, 6^3=5, 8^3=11, 10^3=9, 12^3=15, 14^3=13.
// Differences: -2, 6, -2, 6, -2, 6, -2 → non-uniform → NonAffine → gather.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @xor_pattern
// CHECK: vector.gather
// CHECK-NOT: vector.transfer_read
func.func @xor_pattern(%A: memref<?xf64>, %B: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c3 = arith.constant 3 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %shifted = arith.shli %i, %c1 : index
    %xored = arith.xori %shifted, %c3 : index
    %v = memref.load %A[%xored] : memref<?xf64>
    memref.store %v, %B[%i] : memref<?xf64>
  }
  return
}

// -----

// -----

// ---------------------------------------------------------------------------
// Test 5: Two indirects from same array: B[i] = A[idx[i]] + A[idx[i+1]]
//
// idx[i] is unit-stride → vector.transfer_read for the index load.
// A[idx[i]] and A[idx[i+1]] are NonAffine (runtime address depends on idx
// values) → both become vector.gather.
// The store B[i] is unit-stride → vector.transfer_write.
//
// Verifies that when TWO NonAffine loads target the same source array but with
// different index sub-vectors (idx[i..i+7] vs idx[i+1..i+8]), the vectorizer
// emits a separate vector.gather for each — not one gather reused for both.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @two_indirects
// CHECK-COUNT-2: vector.gather
func.func @two_indirects(%A: memref<?xf64>, %idx: memref<?xindex>, %B: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %i1 = arith.addi %i, %c1 : index
    %a_idx = memref.load %idx[%i] : memref<?xindex>
    %b_idx = memref.load %idx[%i1] : memref<?xindex>
    %a = memref.load %A[%a_idx] : memref<?xf64>
    %b = memref.load %A[%b_idx] : memref<?xf64>
    %s = arith.addf %a, %b : f64
    memref.store %s, %B[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 6: @cpu_kernel(grid, tile) Morton-like: outer loop tile_id is invariant
// w.r.t. inner loop; inner addr = tile_id * TILE + local_i (unit-stride w.r.t.
// local_i). Task 1 fix: outer block arg (tile_id) treated as loop-invariant →
// coeff = 1 (unit stride) → vector.transfer_read, NOT vector.gather.
//
// This is the key regression test for the Task-1 fix: before the fix, tile_id
// was classified NonAffine (block arg not recognized as invariant), causing the
// inner loop to emit vector.gather instead of vector.transfer_read.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @cpu_kernel_proxy_morton
// CHECK: vector.transfer_read
// CHECK-NOT: vector.gather
func.func @cpu_kernel_proxy_morton(%A: memref<?xf32>, %B: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  scf.for %tile_id = %c0 to %c32 step %c1 {
    scf.for %local = %c0 to %c8 step %c1 {
      %tile_size = arith.constant 8 : index
      %base = arith.muli %tile_id, %tile_size : index
      %i = arith.addi %base, %local : index
      %v = memref.load %A[%i] : memref<?xf32>
      memref.store %v, %B[%i] : memref<?xf32>
    }
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 7: divui/remui brick pattern (same computation as cross_brick_stencil
// in lego_vectorize.mlir, but with boundary at z=6 (mod 7) creating a
// non-standard cross-block case with > 1 jump → NonAffine → gather).
// addr(z) = (z+1)/7*13 + (z+1)%7  for z=0..7
// z=0: 1%7=1, z=5: 6%7=6, z=6: 7/7*13+0=13 → jump at z=6 → then z=7: 8%7*1...
// Probing z=0..7 reveals: differences = 1,1,1,1,1,7,1 (boundary at idx=6 of diffs).
// Actually boundary is at position 6 in the 8-element probe → CrossBlock(6).
// CHECK-LABEL: func.func @brick_boundary_at_6
// CHECK: vector.transfer_read
// CHECK: vector.shuffle
func.func @brick_boundary_at_6(%A: memref<?xf64>, %B: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c13 = arith.constant 13 : index
  scf.for %z = %c0 to %c8 step %c1 {
    %zp1 = arith.addi %z, %c1 : index
    %bk = arith.divui %zp1, %c7 : index
    %in = arith.remui %zp1, %c7 : index
    %boff = arith.muli %bk, %c13 : index
    %total = arith.addi %in, %boff : index
    %v = memref.load %A[%total] : memref<?xf64>
    memref.store %v, %B[%z] : memref<?xf64>
  }
  return
}
