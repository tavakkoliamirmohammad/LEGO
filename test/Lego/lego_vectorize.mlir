// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// CHECK-LABEL: func.func @noop_passthrough
// CHECK: arith.addi
// CHECK: return
func.func @noop_passthrough(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  return %c : i32
}

// -----

// A trivially unit-stride access: addr = iv (each f64 = 8 bytes).
// AVX-512 default → L_strip = 64/8 = 8 → vector<8xf64>.
// Task 8 emits a vector loop + scalar tail; both must appear in output.
// CHECK-LABEL: func.func @row_major_unit_stride
// CHECK: vector.transfer_read {{.*}} : memref<1024xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<1024xf64>
// CHECK: return
func.func @row_major_unit_stride(%A: memref<1024xf64>, %B: memref<1024xf64>) {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1024 step %c1 {
    %v = memref.load %A[%i] : memref<1024xf64>
    memref.store %v, %B[%i] : memref<1024xf64>
  }
  return
}

// -----

// SAXPY-style kernel: y[i] = a*x[i] + y[i]. Vectorized at L=8 for AVX-512 f64.
// CHECK-LABEL: func.func @saxpy
// CHECK: vector.broadcast {{.*}} : f64 to vector<8xf64>
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: arith.mulf {{.*}} : vector<8xf64>
// CHECK: arith.addf {{.*}} : vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @saxpy(%a: f64, %X: memref<?xf64>, %Y: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %xi = memref.load %X[%i] : memref<?xf64>
    %yi = memref.load %Y[%i] : memref<?xf64>
    %p  = arith.mulf %a, %xi : f64
    %s  = arith.addf %p, %yi : f64
    memref.store %s, %Y[%i] : memref<?xf64>
  }
  return
}

// -----

// Mixed precision: load f32, extf to f64, accumulate into f64.
// L_strip = lcm(16, 8) = 16. f32 load at width 16, extf produces TWO f64
// sub-vectors at width 8 each, then two f64 stores.
// CHECK-LABEL: func.func @mixed_precision
// CHECK: vector.transfer_read {{.*}} : memref<?xf32>, vector<16xf32>
// CHECK: vector.extract_strided_slice {{.*}} {offsets = [0]
// CHECK: vector.extract_strided_slice {{.*}} {offsets = [8]
// CHECK: arith.extf {{.*}} : vector<8xf32> to vector<8xf64>
// CHECK: arith.extf {{.*}} : vector<8xf32> to vector<8xf64>
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: arith.addf {{.*}} : vector<8xf64>
// CHECK: arith.addf {{.*}} : vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @mixed_precision(%X: memref<?xf32>, %C: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %xi32 = memref.load %X[%i] : memref<?xf32>
    %xi64 = arith.extf %xi32 : f32 to f64
    %ci = memref.load %C[%i] : memref<?xf64>
    %s = arith.addf %ci, %xi64 : f64
    memref.store %s, %C[%i] : memref<?xf64>
  }
  return
}

// -----

// Cross-brick read: the inner z loop sees an A[addr(z)] pattern where addr
// uses a brick-style layout with brick_size=8 and brick_stride=16 (> within-
// brick element stride 1):
//
//   addr(z) = (z+1)/8 * 16  +  (z+1)%8
//
// Concrete values for z=0..7 (BRICK_SIZE=8, BRICK_STRIDE=16):
//   z=0: (1/8)*16 + (1%8) = 0*16+1 = 1
//   z=1: (2/8)*16 + (2%8) = 0*16+2 = 2
//   ...
//   z=6: (7/8)*16 + (7%8) = 0*16+7 = 7
//   z=7: (8/8)*16 + (8%8) = 1*16+0 = 16
//
// Differences: 1,1,1,1,1,1,9 — six unit steps then one jump of 9.
// Tier-A cannot reduce divui/remui to an affine expression → NonAffine.
// Tier-B probes z=0..7, computes diffs, finds boundaryCount==1 at z=7,
// verifies both segments have unit element-index stride (diff==1), and
// classifies as CrossBlock(boundary=7).
//
// Emission: two vector.transfer_reads (block N at baseIv, block N+1 at
// baseIv+7) + a vector.shuffle that splices lanes [0..6] from block N and
// lane [0] from block N+1.
//
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
