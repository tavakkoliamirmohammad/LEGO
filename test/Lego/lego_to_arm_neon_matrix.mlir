// RUN: lego-opt %s --lego-to-arm-neon | FileCheck %s

// End-to-end pipeline tests for --lego-to-arm-neon.
// At v1, the ARM NEON pipeline is structurally identical to --lego-to-x86-vector:
// lego-vectorize uses AVX-512 defaults (vector<8xf64>). The AArch64 backend
// splits to NEON-width (2xf64 per vector register) during codegen.
// Proper NEON-width vector selection (2 f64 / 4 f32 lanes) is deferred to R15.

// ---------------------------------------------------------------------------
// Case 1: SAXPY (unit-stride, broadcast scalar).
// Expected: llvm.fmul + llvm.fadd on vector<8xf64> (AVX-512 default at v1).
// ---------------------------------------------------------------------------
// CHECK-LABEL: llvm.func @saxpy_neon_matrix
// CHECK-NOT: arith.
// CHECK-NOT: scf.for
// CHECK-NOT: vector.
// CHECK: vector<8xf64>
// CHECK: llvm.fmul
// CHECK: llvm.fadd
// CHECK: llvm.return
func.func @saxpy_neon_matrix(%a: f64, %X: memref<?xf64>, %Y: memref<?xf64>, %N: index) {
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

// ---------------------------------------------------------------------------
// Case 2: Unit-stride copy.
// Expected: vector<8xf64> load + store in LLVM dialect.
// ---------------------------------------------------------------------------
// CHECK-LABEL: llvm.func @unit_copy_neon
// CHECK: vector<8xf64>
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK: llvm.return
func.func @unit_copy_neon(%A: memref<?xf64>, %B: memref<?xf64>, %N: index) {
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
// Case 3: Cross-block shuffle (vector.shuffle → llvm.shufflevector).
// ---------------------------------------------------------------------------
// CHECK-LABEL: llvm.func @cross_block_neon
// CHECK: llvm.shufflevector
// CHECK: llvm.return
func.func @cross_block_neon(%A: memref<?xf64>, %B: memref<?xf64>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  scf.for %z = %c0 to %c8 step %c1 {
    %zp1 = arith.addi %z, %c1 : index
    %bk = arith.divui %zp1, %c8 : index
    %in = arith.remui %zp1, %c8 : index
    %boff = arith.muli %bk, %c16 : index
    %total = arith.addi %in, %boff : index
    %v = memref.load %A[%total] : memref<?xf64>
    memref.store %v, %B[%z] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Case 4: Morton gather.
// Expected: index vector computation + vector store in LLVM output.
// ---------------------------------------------------------------------------
// CHECK-LABEL: llvm.func @morton_neon
// CHECK: vector<8xf64>
// CHECK: llvm.return
func.func @morton_neon(%A: memref<?xf64>, %B: memref<?xf64>, %outer: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c5 = arith.constant 5 : index
  scf.for %j = %c0 to %c8 step %c1 {
    %ti = arith.andi %outer, %c5 : index
    %tj = arith.andi %j, %c5 : index
    %tj2 = arith.shli %tj, %c1 : index
    %morton = arith.ori %ti, %tj2 : index
    %v = memref.load %A[%morton] : memref<?xf64>
    memref.store %v, %B[%j] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Case 5: Mixed-precision f32→f64.
// Expected: vector<8xf64> fadd in LLVM output.
// ---------------------------------------------------------------------------
// CHECK-LABEL: llvm.func @mixed_f32_f64_neon
// CHECK: vector<8xf64>
// CHECK: llvm.fadd
// CHECK: llvm.return
func.func @mixed_f32_f64_neon(%X: memref<?xf32>, %C: memref<?xf64>, %N: index) {
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

// ---------------------------------------------------------------------------
// Case 6: RAW-blocked self-update should NOT vectorize.
// Scalar llvm.load/store (no vector<8xf64>) should survive.
// ---------------------------------------------------------------------------
// CHECK-LABEL: llvm.func @self_update_neon
// CHECK-NOT: vector<8xf64>
// CHECK: llvm.load
// CHECK: llvm.store
// CHECK: llvm.return
func.func @self_update_neon(%A: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c1 to %N step %c1 {
    %im1 = arith.subi %i, %c1 : index
    %prev = memref.load %A[%im1] : memref<?xf64>
    %cur  = memref.load %A[%i]   : memref<?xf64>
    %s = arith.addf %prev, %cur : f64
    memref.store %s, %A[%i] : memref<?xf64>
  }
  return
}
