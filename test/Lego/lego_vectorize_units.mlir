// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// Comprehensive unit-stride access pattern tests.
// All tests target AVX-512 default: L_strip = 64/elementBytes.
// For f64: L=8 lanes. For f32: L=16 lanes.

// ---------------------------------------------------------------------------
// Test 1: 1D contiguous load+store (basic unit-stride).
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @unit_1d_basic
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @unit_1d_basic(%A: memref<?xf64>, %B: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %A[%i] : memref<?xf64>
    memref.store %v, %B[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 2: Multiple unit accesses A→B and A→C in same loop.
// Both stores must get vectorized.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @unit_two_stores
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @unit_two_stores(%A: memref<?xf64>, %B: memref<?xf64>, %C: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %A[%i] : memref<?xf64>
    memref.store %v, %B[%i] : memref<?xf64>
    memref.store %v, %C[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 3: Mixed read+write of different bases (A→B, B→C in consecutive loops).
// Each loop should independently vectorize.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @unit_two_loops
// First loop: A→B
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
// Second loop: B→C
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @unit_two_loops(%A: memref<?xf64>, %B: memref<?xf64>, %C: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %A[%i] : memref<?xf64>
    memref.store %v, %B[%i] : memref<?xf64>
  }
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %B[%i] : memref<?xf64>
    memref.store %v, %C[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 4: Static trip count divisible by L=8 (64 elements).
// Vec loop covers all iterations; tail loop body is empty (will be emitted
// but has the form: scf.for %i = alignedUb to %c64 step %c1 { ... })
// alignedUb == 64 so tail never executes.
// Both vec loop and tail loop always emitted (tail body empty at runtime).
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @unit_divisible_trip
// CHECK: vector.transfer_read {{.*}} : memref<64xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<64xf64>
// Tail loop is emitted but its trip count is 0 at runtime (64 % 8 == 0).
// CHECK: scf.for
func.func @unit_divisible_trip(%A: memref<64xf64>, %B: memref<64xf64>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c64 step %c1 {
    %v = memref.load %A[%i] : memref<64xf64>
    memref.store %v, %B[%i] : memref<64xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 5: Static trip count NOT divisible by L (100 elements, 100 = 8*12+4).
// Expect both a vec loop (covering 96 iterations) and a scalar tail (4 iters).
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @unit_tail_trip
// CHECK: vector.transfer_read {{.*}} : memref<100xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<100xf64>
// CHECK: memref.load {{.*}} : memref<100xf64>
// CHECK: memref.store {{.*}} : memref<100xf64>
func.func @unit_tail_trip(%A: memref<100xf64>, %B: memref<100xf64>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c100 step %c1 {
    %v = memref.load %A[%i] : memref<100xf64>
    memref.store %v, %B[%i] : memref<100xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 6: Dynamic trip count (N is a function argument).
// Both vec loop and scalar tail must appear.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @unit_dynamic_trip
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
// CHECK: memref.load {{.*}} : memref<?xf64>
// CHECK: memref.store {{.*}} : memref<?xf64>
func.func @unit_dynamic_trip(%A: memref<?xf64>, %B: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %A[%i] : memref<?xf64>
    memref.store %v, %B[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 7: f32 unit-stride; L=16 (AVX-512, 64/4 = 16 f32 lanes).
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @unit_f32
// CHECK: vector.transfer_read {{.*}} : memref<?xf32>, vector<16xf32>
// CHECK: vector.transfer_write {{.*}} : vector<16xf32>, memref<?xf32>
func.func @unit_f32(%A: memref<?xf32>, %B: memref<?xf32>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %A[%i] : memref<?xf32>
    memref.store %v, %B[%i] : memref<?xf32>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 8: i32 unit-stride; L=16 (64-byte register / 4 bytes = 16 lanes).
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @unit_i32
// CHECK: vector.transfer_read {{.*}} : memref<?xi32>, vector<16xi32>
// CHECK: vector.transfer_write {{.*}} : vector<16xi32>, memref<?xi32>
func.func @unit_i32(%A: memref<?xi32>, %B: memref<?xi32>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %A[%i] : memref<?xi32>
    memref.store %v, %B[%i] : memref<?xi32>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 9: Unit-stride with arith operations in the loop body.
// Verify arith ops are emitted at vector width.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @unit_with_arith
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: arith.mulf {{.*}} : vector<8xf64>
// CHECK: arith.addf {{.*}} : vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @unit_with_arith(%A: memref<?xf64>, %B: memref<?xf64>, %C: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %a = memref.load %A[%i] : memref<?xf64>
    %b = memref.load %B[%i] : memref<?xf64>
    %p = arith.mulf %a, %b : f64
    %s = arith.addf %p, %b : f64
    memref.store %s, %C[%i] : memref<?xf64>
  }
  return
}
