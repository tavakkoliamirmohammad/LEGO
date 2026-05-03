// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// Strided access pattern tests.
// Strided accesses (stride > 1) are emitted as vector.gather by the vectorizer.
// TierA classifies: coeff > 1 with constant factor → Strided.
// TierB classifies: uniform non-unit differences → Strided.

// ---------------------------------------------------------------------------
// Test 1: Column-major access in a row loop — stride is a constant value.
// addr(i) = i * 64: constant stride of 64 elements → Strided.
// Emitted as vector.gather with index vector [base, base+64, base+128, ...].
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @col_major_static_stride
// CHECK: vector.gather
// CHECK-NOT: vector.transfer_read
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
// Test 2: Stride that is a runtime value — TierA yields NonAffine (muli by
// non-constant). TierB probes concrete values but since N is a function arg
// (not evaluable), falls back to NonAffine → also emits vector.gather.
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
// Test 3: Stride exactly equal to register-width-in-bytes (8 elements = 64 bytes
// for f64). Still a constant stride greater than 1 element → Strided → gather.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @stride_equals_regwidth
// CHECK: vector.gather
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
// Test 4: Two strided accesses in the same loop with different strides.
// addr_a(i) = i*4, addr_b(i) = i*16.
// Both loads emit vector.gather; the unit-stride store gets transfer_write.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @two_strided_accesses
// CHECK: vector.gather
// CHECK: vector.gather
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
// A[i] is unit-stride → transfer_read.
// B[i*4] is strided → gather.
// The mixed case still vectorizes: unit-stride accesses are present (hasUnit=true),
// so the cost-penalty is not applied. Result: transfer_read + gather + transfer_write.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @mixed_unit_strided
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: vector.gather
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
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
