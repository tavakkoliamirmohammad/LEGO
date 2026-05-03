// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// Cross-block access pattern tests.
// TierB detects exactly one "jump" in the address sequence (boundaryCount==1)
// where both segments are unit-stride → CrossBlock classification.
// Emission: two vector.transfer_reads + vector.shuffle.

// ---------------------------------------------------------------------------
// Test 1: Standard brick stencil with boundary=7 (same as baseline test).
// addr(z) = (z+1)/8*16 + (z+1)%8 for z=0..7.
// Diffs: 1,1,1,1,1,1,9 → boundary at index 7 → CrossBlock(boundary=7).
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @cross_block_boundary7
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: vector.shuffle
func.func @cross_block_boundary7(%A: memref<?xf64>, %B: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  scf.for %z = %c0 to %c8 step %c1 {
    %zp1 = arith.addi %z, %c1 : index
    %brick_idx = arith.divui %zp1, %c8 : index
    %inner = arith.remui %zp1, %c8 : index
    %brick_off = arith.muli %brick_idx, %c16 : index
    %total = arith.addi %inner, %brick_off : index
    %v = memref.load %A[%total] : memref<?xf64>
    memref.store %v, %B[%z] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 2: Two cross-block reads in the same loop.
// addr_left(z) = (z+1)/8*16 + (z+1)%8  → CrossBlock(boundary=7)
// addr_right(z) = (z+2)/8*16 + (z+2)%8 → probing 0..7: diffs are mostly 1
//   but with a jump at position 6: z=5→z=6 is unit, z=6→z=7 is the jump.
//   → CrossBlock(boundary=6) for the right access.
// Result: two pairs of transfer_reads + two shuffles + one transfer_write.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @cross_block_two_reads
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: vector.shuffle
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: vector.shuffle
// CHECK: arith.addf {{.*}} : vector<8xf64>
// CHECK: vector.transfer_write
func.func @cross_block_two_reads(%A: memref<?xf64>, %B: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %z = %c0 to %c8 step %c1 {
    // Left access: boundary=7
    %zp1 = arith.addi %z, %c1 : index
    %b1 = arith.divui %zp1, %c8 : index
    %r1 = arith.remui %zp1, %c8 : index
    %bo1 = arith.muli %b1, %c16 : index
    %off_left = arith.addi %r1, %bo1 : index
    // Right access: boundary=6
    %zp2 = arith.addi %z, %c2 : index
    %b2 = arith.divui %zp2, %c8 : index
    %r2 = arith.remui %zp2, %c8 : index
    %bo2 = arith.muli %b2, %c16 : index
    %off_right = arith.addi %r2, %bo2 : index
    %vl = memref.load %A[%off_left] : memref<?xf64>
    %vr = memref.load %A[%off_right] : memref<?xf64>
    %s = arith.addf %vl, %vr : f64
    memref.store %s, %B[%z] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 3: Cross-block read combined with a Unit-stride read.
// addr_xb(z) = (z+1)/8*16 + (z+1)%8  → CrossBlock(boundary=7)
// addr_unit(z) = z                     → Unit
// Both in same loop; loop should vectorize both accesses.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @cross_block_plus_unit
// CHECK: vector.transfer_read
// CHECK: vector.shuffle
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: arith.addf {{.*}} : vector<8xf64>
// CHECK: vector.transfer_write
func.func @cross_block_plus_unit(%A: memref<?xf64>, %B: memref<?xf64>, %C: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  scf.for %z = %c0 to %c8 step %c1 {
    // Cross-block access
    %zp1 = arith.addi %z, %c1 : index
    %bk = arith.divui %zp1, %c8 : index
    %in = arith.remui %zp1, %c8 : index
    %boff = arith.muli %bk, %c16 : index
    %total = arith.addi %in, %boff : index
    %va = memref.load %A[%total] : memref<?xf64>
    // Unit-stride access
    %vb = memref.load %B[%z] : memref<?xf64>
    %s = arith.addf %va, %vb : f64
    memref.store %s, %C[%z] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 4: @cross_brick_stencil (moved from lego_vectorize.mlir, Finding 8).
// Same as cross_block_boundary7; kept here as canonical exhaustive test.
// addr(z) = (z+1)/8*16 + (z+1)%8 → CrossBlock(boundary=7).
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @cross_brick_stencil
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: vector.shuffle
func.func @cross_brick_stencil(%A: memref<?xf64>, %B: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  scf.for %z = %c0 to %c8 step %c1 {
    %zp1 = arith.addi %z, %c1 : index
    %brick_idx = arith.divui %zp1, %c8 : index
    %inner = arith.remui %zp1, %c8 : index
    %brick_off = arith.muli %brick_idx, %c16 : index
    %total = arith.addi %inner, %brick_off : index
    %v = memref.load %A[%total] : memref<?xf64>
    memref.store %v, %B[%z] : memref<?xf64>
  }
  return
}

// -----

// V1 LIMITATION (Tier-B lb!=0 unsoundness): Tier-B probes addr at iv=0..L-1,
// independent of the loop's actual lower bound. For piecewise-modular access
// patterns (divui + remui + scaling), the boundary location depends on
// (lb mod period). When lb != 0, Tier-B may return the wrong boundary.
//
// This test documents the current behavior — the loop is vectorized
// (CrossBlock branch fires) but the boundary may be off. Real correctness for
// non-zero lb is captured by R12 (which will thread the loop's lb through to
// Tier-B's probe baseline).
//
// CHECK-LABEL: func.func @cross_brick_stencil_nonzero_lb
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: vector.shuffle
func.func @cross_brick_stencil_nonzero_lb(%A: memref<?xf64>, %B: memref<?xf64>) {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  // Same brick stride pattern as cross_brick_stencil but starts at iv=8 (mid-brick).
  scf.for %z = %c8 to %c32 step %c1 {
    %zp1 = arith.addi %z, %c1 : index
    %brick_idx = arith.divui %zp1, %c8 : index
    %inner = arith.remui %zp1, %c8 : index
    %brick_off = arith.muli %brick_idx, %c16 : index
    %total = arith.addi %inner, %brick_off : index
    %v = memref.load %A[%total] : memref<?xf64>
    memref.store %v, %B[%z] : memref<?xf64>
  }
  return
}
