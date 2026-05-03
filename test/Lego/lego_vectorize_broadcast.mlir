// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// Broadcast (loop-invariant scalar) tests.
// A Broadcast access has coeff=0 in TierA (address is loop-invariant).
// Pure-broadcast loops (only Broadcast accesses, no Unit) are NOT vectorized
// (sawConstraining = false). Loops with Unit + Broadcast vectorize both.

// ---------------------------------------------------------------------------
// Test 1: Loop-invariant scalar argument in SAXPY-style expression.
// %a is a function argument → outside the loop → broadcastMap entry.
// Expected: vector.broadcast %a : f64 to vector<8xf64>
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @scalar_arg_broadcast
// CHECK: vector.broadcast {{.*}} : f64 to vector<8xf64>
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: arith.mulf {{.*}} : vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @scalar_arg_broadcast(%a: f64, %X: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %X[%i] : memref<?xf64>
    %s = arith.mulf %a, %v : f64
    memref.store %s, %X[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 2: Loop-invariant memref load outside the loop (pre-hoisted scalar).
// The load result is a scalar used inside the loop → broadcast.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @prehoisted_load_broadcast
// CHECK: memref.load {{.*}} : memref<?xf64>
// CHECK: vector.broadcast {{.*}} : f64 to vector<8xf64>
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: arith.mulf {{.*}} : vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @prehoisted_load_broadcast(%A: memref<?xf64>, %B: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %scalar = memref.load %B[%c5] : memref<?xf64>
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %A[%i] : memref<?xf64>
    %s = arith.mulf %v, %scalar : f64
    memref.store %s, %A[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 3: Loop-invariant load inside the loop (classified as Broadcast by
// TierA: index = constant → coeff=0 → Broadcast kind).
// Emitted as: scalar load + vector.broadcast in the vectorized body.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @inloop_invariant_load
// CHECK: memref.load {{.*}} : memref<?xf64>
// CHECK: vector.broadcast {{.*}} : f64 to vector<8xf64>
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: arith.addf {{.*}} : vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @inloop_invariant_load(%A: memref<?xf64>, %B: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  scf.for %i = %c0 to %N step %c1 {
    %bval = memref.load %B[%c10] : memref<?xf64>
    %v = memref.load %A[%i] : memref<?xf64>
    %s = arith.addf %v, %bval : f64
    memref.store %s, %A[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 4: Multiple broadcasts in same loop (two scalars: %a and %b).
// Each scalar should get a separate vector.broadcast op.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @multi_broadcast
// CHECK: vector.broadcast {{.*}} : f64 to vector<8xf64>
// CHECK: vector.broadcast {{.*}} : f64 to vector<8xf64>
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
func.func @multi_broadcast(%a: f64, %b: f64, %X: memref<?xf64>, %Y: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %X[%i] : memref<?xf64>
    %pa = arith.mulf %a, %v : f64
    %pb = arith.mulf %b, %pa : f64
    memref.store %pb, %Y[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 5: Pure-broadcast loop (only a constant-index load, no unit stride).
// sawConstraining = false → L_strip = 1 → NOT vectorized.
// No vector ops should appear.
// TODO(R12): Future: detect pure-broadcast loops and decide whether to hoist.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @pure_broadcast_no_vec
// CHECK-NOT: vector.transfer_read
// CHECK-NOT: vector.broadcast
// CHECK: memref.load
// CHECK: memref.store
func.func @pure_broadcast_no_vec(%A: memref<?xf64>, %B: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  scf.for %i = %c0 to %N step %c1 {
    %bval = memref.load %B[%c5] : memref<?xf64>
    memref.store %bval, %A[%c5] : memref<?xf64>
  }
  return
}
