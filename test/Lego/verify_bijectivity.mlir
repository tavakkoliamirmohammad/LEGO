// RUN: lego-opt -lego-verify-bijectivity %s -verify-diagnostics -split-input-file

// Test 1: Row-major layout (should pass - it's bijective)
func.func @test_row_major_bijective() {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Row-major: flat = i * 8 + j
  // Inverse: i = flat / 8, j = flat % 8
  %layout = lego.gen_p [%c4, %c8] apply (%i: index, %j: index) {
    %c8_apply = arith.constant 8 : index
    %t = arith.muli %i, %c8_apply : index
    %flat = arith.addi %t, %j : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c8_inv = arith.constant 8 : index
    %i_out = arith.divui %flat, %c8_inv : index
    %j_out = arith.remui %flat, %c8_inv : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  return
}

// -----

// Test 2: Column-major layout (should pass - it's bijective)
func.func @test_col_major_bijective() {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Column-major: flat = j * 4 + i
  // Inverse: j = flat / 4, i = flat % 4
  %layout = lego.gen_p [%c4, %c8] apply (%i: index, %j: index) {
    %c4_apply = arith.constant 4 : index
    %t = arith.muli %j, %c4_apply : index
    %flat = arith.addi %t, %i : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c4_inv = arith.constant 4 : index
    %j_out = arith.divui %flat, %c4_inv : index
    %i_out = arith.remui %flat, %c4_inv : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  return
}

// -----

// Test 3: Non-bijective layout - wrong inverse (should fail)
func.func @test_non_bijective_wrong_inv() {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Apply: flat = i * 8 + j (row-major)
  // Inverse: WRONG - always returns (0, 0)
  // expected-error@+1 {{Layout is not bijective}}
  %layout = lego.gen_p [%c4, %c8] apply (%i: index, %j: index) {
    %c8_apply = arith.constant 8 : index
    %t = arith.muli %i, %c8_apply : index
    %flat = arith.addi %t, %j : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %zero = arith.constant 0 : index
    lego.yield %zero, %zero : index, index
  } : !lego.layout

  return
}

// -----

// Test 4: Non-bijective layout - swapped dimensions (should fail)
func.func @test_non_bijective_swapped() {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Apply: flat = i * 8 + j
  // Inverse: Returns (j, i) instead of (i, j) - dimensions swapped
  // expected-error@+1 {{Layout is not bijective}}
  %layout = lego.gen_p [%c4, %c8] apply (%i: index, %j: index) {
    %c8_apply = arith.constant 8 : index
    %t = arith.muli %i, %c8_apply : index
    %flat = arith.addi %t, %j : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c8_inv = arith.constant 8 : index
    %i_computed = arith.divui %flat, %c8_inv : index
    %j_computed = arith.remui %flat, %c8_inv : index
    // Swap the outputs!
    lego.yield %j_computed, %i_computed : index, index
  } : !lego.layout

  return
}

// -----

// Test 5: Non-surjective apply (should fail for apply∘inv check)
func.func @test_non_surjective_apply() {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  // Apply: Always returns flat index 0 (not injective or surjective)
  // Inverse: Standard row-major inverse
  // expected-error@+1 {{Layout is not bijective}}
  %layout = lego.gen_p [%c4, %c8] apply (%i: index, %j: index) {
    %zero = arith.constant 0 : index
    lego.yield %zero : index
  } inv (%flat: index) {
    %c8_inv = arith.constant 8 : index
    %i_out = arith.divui %flat, %c8_inv : index
    %j_out = arith.remui %flat, %c8_inv : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  return
}

// -----

// Test 6: Z-order (Morton) layout (should pass - it's bijective for power-of-2 dims)
func.func @test_morton_bijective() {
  %c4 = arith.constant 4 : index

  // Simplified Morton encoding for 2x2 bits (4x4 grid)
  // This is a simplified version - full Morton requires bit interleaving
  %layout = lego.gen_p [%c4, %c4] apply (%i: index, %j: index) {
    %c2 = arith.constant 2 : index
    // Simplified: group into 2x2 tiles
    %ti = arith.divui %i, %c2 : index
    %tj = arith.divui %j, %c2 : index
    %li = arith.remui %i, %c2 : index
    %lj = arith.remui %j, %c2 : index

    %c4_mul = arith.constant 4 : index
    %tile_idx = arith.muli %ti, %c2 : index
    %tile_flat = arith.addi %tile_idx, %tj : index
    %tile_offset = arith.muli %tile_flat, %c4_mul : index

    %local_flat = arith.muli %li, %c2 : index
    %local_idx = arith.addi %local_flat, %lj : index

    %flat = arith.addi %tile_offset, %local_idx : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c2 = arith.constant 2 : index
    %c4_div = arith.constant 4 : index

    %tile_flat = arith.divui %flat, %c4_div : index
    %local_idx = arith.remui %flat, %c4_div : index

    %ti = arith.divui %tile_flat, %c2 : index
    %tj = arith.remui %tile_flat, %c2 : index

    %li = arith.divui %local_idx, %c2 : index
    %lj = arith.remui %local_idx, %c2 : index

    %i_base = arith.muli %ti, %c2 : index
    %i_out = arith.addi %i_base, %li : index

    %j_base = arith.muli %tj, %c2 : index
    %j_out = arith.addi %j_base, %lj : index

    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  return
}
