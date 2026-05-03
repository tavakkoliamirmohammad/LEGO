// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// R17 (partial): scf.if predicated maskedstore.
// Supports:
//   scf.if %cond {
//     memref.store %val, %B[%i]
//   }
// Lowers to: vector.maskedstore with the vectorised condition as mask.
//
// This is the "conditional store" pattern common in data-dependent write kernels:
// copy values from A to B only where a condition holds.

// ---------------------------------------------------------------------------
// Test 1: conditional store with cmpf condition.
// Loop: if A[i] > thresh: B[i] = A[i]
// The condition vector<L x i1> from cmpf gates the maskedstore.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @cond_store
// CHECK: arith.cmpf
// CHECK: vector.maskedstore
func.func @cond_store(%A: memref<?xf64>, %B: memref<?xf64>,
                      %thresh: f64, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %A[%i] : memref<?xf64>
    %cmp = arith.cmpf ogt, %v, %thresh : f64
    scf.if %cmp {
      memref.store %v, %B[%i] : memref<?xf64>
    }
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 2: conditional store with cmpi condition on loop-invariant i32 values.
// The condition is constant-foldable but vectorized correctly regardless.
// Uses two loop-invariant i32 constants (not IV-dependent), so bodyOK passes.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @cond_store_cmpi
// CHECK: arith.cmpi
// CHECK: vector.maskedstore
func.func @cond_store_cmpi(%A: memref<?xf64>, %B: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %ci32_5 = arith.constant 5 : i32
  %ci32_3 = arith.constant 3 : i32
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %A[%i] : memref<?xf64>
    %cmp = arith.cmpi sgt, %ci32_5, %ci32_3 : i32
    scf.if %cmp {
      memref.store %v, %B[%i] : memref<?xf64>
    }
  }
  return
}
