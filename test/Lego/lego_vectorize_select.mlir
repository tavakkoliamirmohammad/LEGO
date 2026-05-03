// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// R16: arith.cmpi / arith.cmpf / arith.select vectorization.
//
// Previously [G1] rejected loops with arith.cmpi/cmpf on index-typed operands.
// R16 removes that restriction:
//   - Non-index cmpi/cmpf: falls through to normal arith catch-all.
//   - Index-typed cmpi: index operands are cast to i64, comparison emitted on i64 vectors.
//   - arith.select on non-index f64 result: works via normal arith catch-all.

// ---------------------------------------------------------------------------
// Test 1: cmpf (non-index) + select on f64. Standard path.
// arith.cmpf produces i1; arith.select picks between f64 values.
// Result: both cmpf and select are vectorized.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @cmpf_select
// CHECK: arith.cmpf {{.*}} : vector<8xf64>
// CHECK: arith.select {{.*}} : vector<8xi1>, vector<8xf64>
func.func @cmpf_select(%A: memref<?xf64>, %B: memref<?xf64>,
                       %thresh: f64, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %A[%i] : memref<?xf64>
    %cmp = arith.cmpf ogt, %v, %thresh : f64
    %sel = arith.select %cmp, %v, %zero : f64
    memref.store %sel, %B[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 2: cmpi on i32 loop-invariant operands.
// thresh is a function arg (loop-invariant scalar). The broadcast pre-pass will
// broadcast it. The loop-invariant %thresh is broadcast; the loaded f64 value is
// vector<8xf64>; the cmpi uses them.
// Note: arith.index_cast on the IV is NOT included here to keep the test clean.
// Instead we use a pre-cast invariant.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @cmpi_select_f64
// CHECK: arith.cmpi {{.*}}
// CHECK: arith.select
func.func @cmpi_select_f64(%A: memref<?xf64>, %B: memref<?xf64>,
                           %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %one  = arith.constant 1.0 : f64
  %ci32_5 = arith.constant 5 : i32
  %ci32_3 = arith.constant 3 : i32
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %A[%i] : memref<?xf64>
    %cmp = arith.cmpi sgt, %ci32_5, %ci32_3 : i32
    %sel = arith.select %cmp, %v, %zero : f64
    memref.store %sel, %B[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 3: cmpi on index-typed operands (previously [G1] rejected case).
// Both operands are index (loop bound and constant). These are loop-invariant,
// so bodyOK approves (no IV-dependent index_cast result).
// The R16 index-cmpi path converts them to i64 before comparison.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @cmpi_index_operands
// CHECK: arith.cmpi {{.*}} : vector<8xi64>
func.func @cmpi_index_operands(%A: memref<?xf64>, %B: memref<?xf64>,
                               %bound: index, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %chalf = arith.constant 512 : index
  %zero = arith.constant 0.0 : f64
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %A[%i] : memref<?xf64>
    // Compare two loop-invariant index values (both are function args / constants).
    %cmp = arith.cmpi slt, %bound, %chalf : index
    %sel = arith.select %cmp, %v, %zero : f64
    memref.store %sel, %B[%i] : memref<?xf64>
  }
  return
}
