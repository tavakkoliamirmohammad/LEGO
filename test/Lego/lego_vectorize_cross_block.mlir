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

// -----

// ---------------------------------------------------------------------------
// R12: Multi-boundary CrossBlock — 3D 7-point stencil with TWO brick boundaries.
//
// Brick size = 3, brick stride = 6 (2× brick size, with padding).
// addr(z) = (z / 3) * 6 + (z % 3).
// Probing z=0..7 (L=8 for f64 avx512):
//   z=0: 0, z=1: 1, z=2: 2,
//   z=3: 6 (boundary at pos 3), z=4: 7, z=5: 8,
//   z=6: 12 (boundary at pos 6), z=7: 13.
// Diffs: 1,1,4(jump),1,1,4(jump),1 → 2 boundaries → multi-boundary CrossBlock.
//
// Expected emission: 3 transfer_reads (blocks 0,1,2) + 2 shuffles (chain).
// CHECK-LABEL: func.func @cross_block_two_boundaries
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: vector.shuffle
// CHECK: vector.shuffle
// ---------------------------------------------------------------------------
func.func @cross_block_two_boundaries(%A: memref<?xf64>, %B: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c1 = arith.constant 1 : index
  scf.for %z = %c0 to %c8 step %c1 {
    // Brick layout: brick_size=3, brick_stride=6.
    // addr(z) = (z / 3) * 6 + (z % 3).
    %brick_idx = arith.divui %z, %c3 : index
    %inner     = arith.remui %z, %c3 : index
    %brick_off = arith.muli %brick_idx, %c6 : index
    %total     = arith.addi %inner, %brick_off : index
    %v = memref.load %A[%total] : memref<?xf64>
    memref.store %v, %B[%z] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// R12: 3D 7-point stencil simulation — three independent CrossBlock reads.
//
// Models a vectorized inner-z loop of a 3D stencil in brick layout where:
//   center A[flat]: unit-stride (simplest case)
//   A[flat - 1] : CrossBlock (same brick, but position 0 of each brick
//                  wraps to the previous brick's last element)
//   A[flat + 1] : CrossBlock (same boundary issue on the high side)
//
// Here we use a small brick size to force multi-boundary access.
// Brick size=4, brick stride=8. For L=8 lanes:
//   A[flat - 1]: addr(z) = (z+3)/4*8 + (z+3)%4 - 4
//                         for z=0..7 probing: varies with one boundary.
//   A[flat + 1]: addr(z) = (z+5)/4*8 + (z+5)%4 - 4.
//
// This test verifies that EACH CrossBlock access independently emits its
// own (M+1 reads + M shuffles) sequence, producing multiple shuffle ops total.
//
// CHECK-LABEL: func.func @cross_block_3d7pt_simulation
// CHECK: vector.shuffle
// CHECK: vector.shuffle
// ---------------------------------------------------------------------------
func.func @cross_block_3d7pt_simulation(%A: memref<?xf64>, %B: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c8s = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %z = %c0 to %c8 step %c1 {
    // Left neighbor: addr = (z+4)/4*8 + (z+4)%4 — CrossBlock with boundary.
    %zp4        = arith.addi %z, %c4 : index
    %bk_l = arith.divui %zp4, %c4 : index
    %in_l = arith.remui %zp4, %c4 : index
    %bo_l = arith.muli %bk_l, %c8s : index
    %left_addr = arith.addi %in_l, %bo_l : index

    // Right neighbor: addr = (z+1)/4*8 + (z+1)%4 — CrossBlock with boundary.
    %zp1a       = arith.addi %z, %c1 : index
    %bk_r = arith.divui %zp1a, %c4 : index
    %in_r = arith.remui %zp1a, %c4 : index
    %bo_r = arith.muli %bk_r, %c8s : index
    %right_addr = arith.addi %in_r, %bo_r : index

    // Center: unit-stride (z maps to flat index directly).
    %vc = memref.load %A[%z]           : memref<?xf64>
    %vl = memref.load %A[%left_addr]   : memref<?xf64>
    %vr = memref.load %A[%right_addr]  : memref<?xf64>
    %s1 = arith.addf %vc, %vl : f64
    %s2 = arith.addf %s1, %vr : f64
    memref.store %s2, %B[%z] : memref<?xf64>
  }
  return
}
