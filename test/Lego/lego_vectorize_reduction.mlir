// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// R18b: iter_args associative reduction vectorization.
// Loops with a single iter_arg forming an associative reduction (addf, mulf,
// addi, etc.) are vectorized by replacing the scalar accumulator with a
// vector<L x T> accumulator and emitting vector.reduction after the loop.

// ---------------------------------------------------------------------------
// Test 1: Dot-product (addf + mulf chain).  f32 on avx512 → vector<16xf32>.
// The k-loop carries acc = addf(acc, mulf(A[k], B[k])).
// Expected:
//   - vector<16xf32> iter_arg for the vec loop
//   - arith.addf ... : vector<16xf32>   (element-wise accumulation)
//   - vector.reduction <add> ... : vector<16xf32> into f32
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @reduction_dot_product
// CHECK: vector<16xf32>
// CHECK: arith.addf {{.*}} : vector<16xf32>
// CHECK: vector.reduction <add>
func.func @reduction_dot_product(%A: memref<?xf32>, %B: memref<?xf32>, %N: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %sum = scf.for %k = %c0 to %N step %c1 iter_args(%acc = %zero) -> f32 {
    %a = memref.load %A[%k] : memref<?xf32>
    %b = memref.load %B[%k] : memref<?xf32>
    %p = arith.mulf %a, %b : f32
    %s = arith.addf %acc, %p : f32
    scf.yield %s : f32
  }
  return %sum : f32
}

// -----

// ---------------------------------------------------------------------------
// Test 2: Sum-reduction only (no multiply).  f64 on avx512 → vector<8xf64>.
// CHECK-LABEL: func.func @reduction_sum_f64
// CHECK: vector<8xf64>
// CHECK: arith.addf {{.*}} : vector<8xf64>
// CHECK: vector.reduction <add>
// ---------------------------------------------------------------------------
func.func @reduction_sum_f64(%A: memref<?xf64>, %N: index) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f64
  %total = scf.for %k = %c0 to %N step %c1 iter_args(%acc = %zero) -> f64 {
    %a = memref.load %A[%k] : memref<?xf64>
    %s = arith.addf %acc, %a : f64
    scf.yield %s : f64
  }
  return %total : f64
}

// -----

// ---------------------------------------------------------------------------
// Test 3: Integer sum-reduction.  i32 on avx512 → vector<16xi32>.
// CHECK-LABEL: func.func @reduction_sum_i32
// CHECK: vector<16xi32>
// CHECK: arith.addi {{.*}} : vector<16xi32>
// CHECK: vector.reduction <add>
// ---------------------------------------------------------------------------
func.func @reduction_sum_i32(%A: memref<?xi32>, %N: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %total = scf.for %k = %c0 to %N step %c1 iter_args(%acc = %zero) -> i32 {
    %a = memref.load %A[%k] : memref<?xi32>
    %s = arith.addi %acc, %a : i32
    scf.yield %s : i32
  }
  return %total : i32
}
