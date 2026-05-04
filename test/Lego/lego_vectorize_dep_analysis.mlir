// RUN: lego-opt %s --lego-vectorize | FileCheck %s

// Dependence analysis tests.
// The dep analysis (computeMinDepDistance) checks cross-iteration RAW/WAW/WAR
// using affine expressions. If Ld < L_strip, the loop is not vectorized.
//
// (The opt-in --enable-smt-dep-analysis Z3 prover and its SMT- check-prefix
// lines were removed alongside the underlying smtProveNoDep code; per
// feedback_sympy_z3_slow.md the SymPy/Z3 path was avoided in practice.)

// ---------------------------------------------------------------------------
// Test 1: Self-update loop A[i] = A[i-1] + A[i] → RAW cross-iteration dep.
// store index: i (coeff=1, const=0), read index: i-1 (coeff=1, const=-1).
// d = (const_store - const_read) / coeff = (0 - (-1)) / 1 = 1.
// Ld = 1 < L_strip → NOT vectorized.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @self_update_raw
// CHECK-NOT: vector.transfer_read
// CHECK-NOT: vector.gather
// CHECK: memref.load
// CHECK: memref.store
func.func @self_update_raw(%A: memref<?xf64>, %N: index) {
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

// -----

// ---------------------------------------------------------------------------
// Test 2: Provably non-overlapping ranges (read A[i+512], write A[i]).
// store: i (coeff=1, const=0); read: i+512 (coeff=1, const=512).
// d = (0 - 512) / 1 = -512 (negative → WAR, not a fwd dep).
// Ld = max → L_strip = 8 → vectorizes.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @nonoverlap_forward_war
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @nonoverlap_forward_war(%A: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  scf.for %i = %c0 to %N step %c1 {
    %read_idx = arith.addi %i, %c512 : index
    %v = memref.load %A[%read_idx] : memref<?xf64>
    memref.store %v, %A[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 3: Two stores to the same memref at identical expressions (WAW, same
// element every iteration). Affine expressions are identical (coeff, const same)
// → no cross-iteration dep → vectorizes.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @waw_same_element
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @waw_same_element(%A: memref<?xf64>, %B: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %B[%i] : memref<?xf64>
    memref.store %v, %A[%i] : memref<?xf64>
    memref.store %v, %A[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 4: memref.cast root analysis.
// %A_cast : memref<1024xf64> is a cast of %A_dyn : memref<?xf64>.
// Load uses %A_cast, no store. Should vectorize (root of %A_cast is %A_dyn,
// root of %B is %B, roots are disjoint → no dep constraint).
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @cast_root_analysis
// CHECK: vector.transfer_read {{.*}} : memref<1024xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @cast_root_analysis(%A_dyn: memref<?xf64>, %B: memref<?xf64>) {
  %A_cast = memref.cast %A_dyn : memref<?xf64> to memref<1024xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  scf.for %i = %c0 to %c1024 step %c1 {
    %v = memref.load %A_cast[%i] : memref<1024xf64>
    memref.store %v, %B[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 5: memref.subview root analysis.
// %sub : memref<1024xf64> is a subview of %A : memref<?xf64>.
// Load uses %sub, store uses %B (different root). Should vectorize.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @subview_root_analysis
// CHECK: vector.transfer_read {{.*}} : memref<1024xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @subview_root_analysis(%A: memref<?xf64>, %B: memref<?xf64>) {
  %sub = memref.subview %A[0][1024][1] : memref<?xf64> to memref<1024xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  scf.for %i = %c0 to %c1024 step %c1 {
    %v = memref.load %sub[%i] : memref<1024xf64>
    memref.store %v, %B[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 6: RAW with distance > L_strip.
// store index: i (coeff=1, const=0); read index: i+512 with WRITE to A[i].
// i.e. write A[i] = A[i+512] (forward read, backward write for the vector).
// Actually this is the same as Test 2 but let's verify: reading A[i+512]
// and writing A[i] means the write is at a lower address — the read at i+512
// is from a later address. d = (const_store=0 - const_read=512) / 1 = -512.
// Negative d means the stored value is BEFORE the read address (WAR not RAW).
// Ld = max → vectorizes.
// Let's also test a genuine RAW at distance 16 (> L_strip=8):
// write A[i], read A[i-16]. d = (0 - (-16)) / 1 = 16.
// Ld=16 >= L_strip=8 → vectorizes (each strip of 8 doesn't overlap).
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @raw_distance_gt_L
// CHECK: vector.transfer_read {{.*}} : memref<?xf64>, vector<8xf64>
// CHECK: vector.transfer_write {{.*}} : vector<8xf64>, memref<?xf64>
func.func @raw_distance_gt_L(%A: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c16 to %N step %c1 {
    %im16 = arith.subi %i, %c16 : index
    %v = memref.load %A[%im16] : memref<?xf64>
    memref.store %v, %A[%i] : memref<?xf64>
  }
  return
}

// -----

// ---------------------------------------------------------------------------
// Test 7: simple copy loop (disjoint memrefs A and B) — vectorises under
// the default conservative analysis (Ld = max, no cross-iteration dep).
// (Originally also exercised the now-removed --enable-smt-dep-analysis flag.)
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @smt_flag_smoke
// CHECK: vector.transfer_read
// CHECK: vector.transfer_write
func.func @smt_flag_smoke(%A: memref<?xf64>, %B: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %A[%i] : memref<?xf64>
    memref.store %v, %B[%i] : memref<?xf64>
  }
  return
}
