// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// Mixed-precision vectorization tests.
// When accesses have different element widths, L_strip = lcm of their Ln.
// The wider-element load gets split into sub-vectors; narrower elements get
// extended into the wider type.

// ---------------------------------------------------------------------------
// Test 1: f32 → f64 (extension direction, same as baseline test).
// f32: L=16 lanes, f64: L=8 lanes → lcm=16.
// f32 load at width 16 → extract two 8-wide slices → extf each to f64
// → two f64 accumulates → two f64 stores.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @f32_extf_f64
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
func.func @f32_extf_f64(%X: memref<?xf32>, %C: memref<?xf64>, %N: index) {
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
// Test 2: i32 → i64 (integer extension, extsi path).
// i32: L=16 lanes, i64: L=8 lanes → lcm=16.
// Same splitting pattern as f32→f64 but uses arith.extsi.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @i32_extsi_i64
// CHECK: vector.transfer_read {{.*}} : memref<?xi32>, vector<16xi32>
// CHECK: vector.extract_strided_slice {{.*}} {offsets = [0]
// CHECK: vector.extract_strided_slice {{.*}} {offsets = [8]
// CHECK: arith.extsi {{.*}} : vector<8xi32> to vector<8xi64>
// CHECK: arith.extsi {{.*}} : vector<8xi32> to vector<8xi64>
// CHECK: vector.transfer_write {{.*}} : vector<8xi64>, memref<?xi64>
// CHECK: vector.transfer_write {{.*}} : vector<8xi64>, memref<?xi64>
func.func @i32_extsi_i64(%X: memref<?xi32>, %C: memref<?xi64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %xi32 = memref.load %X[%i] : memref<?xi32>
    %xi64 = arith.extsi %xi32 : i32 to i64
    %ci = memref.load %C[%i] : memref<?xi64>
    %s = arith.addi %ci, %xi64 : i64
    memref.store %s, %C[%i] : memref<?xi64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 3: i64 + f64 (integer and float, same width, different types).
// Both have L=8 lanes. arith.sitofp converts i64 vector to f64 vector.
// L_strip=8; both accesses are Unit → vectorizes cleanly.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @i64_sitofp_f64
// CHECK: vector.transfer_read {{.*}} : memref<?xi64>, vector<8xi64>
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: arith.sitofp {{.*}} : vector<8xi64> to vector<8xf64>
// CHECK: arith.addf {{.*}} : vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @i64_sitofp_f64(%X: memref<?xi64>, %Y: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %xi = memref.load %X[%i] : memref<?xi64>
    %yi = memref.load %Y[%i] : memref<?xf64>
    %xf = arith.sitofp %xi : i64 to f64
    %s = arith.addf %xf, %yi : f64
    memref.store %s, %Y[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 4: f32 → f64 with scalar multiplication (SAXPY-like widening loop).
// Similar to Test 1 but with a scalar factor broadcast to f64 width.
// Tests that mixed-precision + scalar broadcast compose correctly.
// ---------------------------------------------------------------------------
// L_strip = lcm(16_f32, 8_f64) = 16. scalar arg broadcast to natural Ln=8
// (one vpbroadcastsd reused for both sub-ops, rather than vector<16xf64>).
// CHECK-LABEL: func.func @f32_extf_saxpy
// CHECK: vector.broadcast {{.*}} : f64 to vector<8xf64>
// CHECK: vector.transfer_read {{.*}} : memref<?xf32>, vector<16xf32>
// CHECK: arith.extf {{.*}} : vector<8xf32> to vector<8xf64>
// CHECK: arith.mulf {{.*}} : vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @f32_extf_saxpy(%a: f64, %X: memref<?xf32>, %C: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %xi32 = memref.load %X[%i] : memref<?xf32>
    %xi64 = arith.extf %xi32 : f32 to f64
    %s = arith.mulf %a, %xi64 : f64
    memref.store %s, %C[%i] : memref<?xf64>
  }
  return
}
