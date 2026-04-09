// RUN: lego-opt -lego-verify-bank-conflicts %s -split-input-file -verify-diagnostics

// Test 1: Conflict-free layout - row-major with 32 columns (should pass)
func.func @test_conflict_free_row_major(%base_tid: index {lego.thread_id}) {
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index

  // Row-major 4x32: Each thread in a warp accesses a different column
  // With 32 banks and 32 columns, each access goes to a different bank
  %layout = lego.gen_p [%c4, %c32] apply (%i: index, %j: index) {
    %c32_apply = arith.constant 32 : index
    %t = arith.muli %i, %c32_apply : index
    %flat = arith.addi %t, %j : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c32_inv = arith.constant 32 : index
    %i_out = arith.divui %flat, %c32_inv : index
    %j_out = arith.remui %flat, %c32_inv : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  // Warp accesses: (0, tid) for tid in [0, 31]
  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----

// Test 2: Bank conflict - column-major with 32 rows (should warn)
func.func @test_conflict_column_major(%base_tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c4 = arith.constant 4 : index

  // Column-major: flat = j * 32 + i
  // When threads access consecutive rows (same column), all access same bank
  %layout = lego.gen_p [%c32, %c4] apply (%i: index, %j: index) {
    %c32_apply = arith.constant 32 : index
    %t = arith.muli %j, %c32_apply : index
    %flat = arith.addi %t, %i : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c32_inv = arith.constant 32 : index
    %j_out = arith.divui %flat, %c32_inv : index
    %i_out = arith.remui %flat, %c32_inv : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  // Warp accesses: (0, tid) - same row across columns
  // Addresses: tid * 32, map to same bank every 32 elements
  %i = arith.constant 0 : index
  %c0_1 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0_1 : index
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----

// Test 3: 2-way bank conflict (should warn)
func.func @test_2way_conflict(%base_tid: index {lego.thread_id}) {
  %c16 = arith.constant 16 : index
  %c2 = arith.constant 2 : index

  // Layout where every two threads access the same bank
  %layout = lego.gen_p [%c16, %c2] apply (%i: index, %j: index) {
    %c2_apply = arith.constant 2 : index
    %c16_apply = arith.constant 16 : index

    // Map to create 2-way conflicts
    // Each pair of consecutive i values maps to same bank
    %i_scaled = arith.muli %i, %c2_apply : index
    %t = arith.muli %j, %c16_apply : index
    %flat = arith.addi %t, %i_scaled : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c2_inv = arith.constant 2 : index
    %c16_inv = arith.constant 16 : index

    %j_out = arith.divui %flat, %c16_inv : index
    %temp = arith.remui %flat, %c16_inv : index
    %i_out = arith.divui %temp, %c2_inv : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  %c0_2 = arith.constant 0 : index
  %i = arith.addi %base_tid, %c0_2 : index
  %j = arith.constant 0 : index
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----

// Test 4: Padded layout to avoid conflicts (should pass)
func.func @test_padded_conflict_free(%base_tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c33 = arith.constant 33 : index

  // 32x33 layout (padded by 1 element per row)
  // This breaks the bank conflict pattern
  %layout = lego.gen_p [%c32, %c33] apply (%i: index, %j: index) {
    %c33_apply = arith.constant 33 : index
    %t = arith.muli %i, %c33_apply : index
    %flat = arith.addi %t, %j : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c33_inv = arith.constant 33 : index
    %i_out = arith.divui %flat, %c33_inv : index
    %j_out = arith.remui %flat, %c33_inv : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  // Consecutive column accesses with padding
  %i = arith.constant 0 : index
  %c0_3 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0_3 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----

// Test 5: Strided access causing conflicts (should warn)
func.func @test_strided_conflicts(%base_tid: index {lego.thread_id}) {
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index

  // Layout where threads access with stride that maps to same banks
  %layout = lego.gen_p [%c64, %c32] apply (%i: index, %j: index) {
    %c32_apply = arith.constant 32 : index
    %t = arith.muli %i, %c32_apply : index
    %flat = arith.addi %t, %j : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c32_inv = arith.constant 32 : index
    %i_out = arith.divui %flat, %c32_inv : index
    %j_out = arith.remui %flat, %c32_inv : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  // Access every other row (stride 2 in row dimension)
  // This creates patterns where addresses map to same banks
  %c2 = arith.constant 2 : index
  %i_scaled = arith.muli %base_tid, %c2 : index
  %j = arith.constant 0 : index
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  %addr = lego.apply %layout(%i_scaled, %j) : !lego.layout

  return
}

// -----

// Test 6: Transpose pattern (likely has conflicts)
func.func @test_transpose_conflicts(%base_tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index

  // Square transpose pattern
  // Reading row-wise, writing column-wise creates bank conflicts
  %layout = lego.gen_p [%c32, %c32] apply (%i: index, %j: index) {
    %c32_apply = arith.constant 32 : index
    // Transpose: swap i and j
    %t = arith.muli %j, %c32_apply : index
    %flat = arith.addi %t, %i : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c32_inv = arith.constant 32 : index
    %j_out = arith.divui %flat, %c32_inv : index
    %i_out = arith.remui %flat, %c32_inv : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  // Access pattern: (0, tid) → column-wise access
  %i = arith.constant 0 : index
  %c0_4 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0_4 : index
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----

// Test 7: Worst-case bank conflict (all threads access different addresses in same bank)
func.func @test_worst_case_conflict(%tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index

  // Layout where all threads map to bank 0 but different addresses
  // flat = tid * 32 (addresses 0, 32, 64, 96, ... all map to bank 0)
  %layout = lego.gen_p [%c64, %c32] apply (%i: index, %j: index) {
    %c32_apply = arith.constant 32 : index
    %flat = arith.muli %i, %c32_apply : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c32_inv = arith.constant 32 : index
    %i_out = arith.divui %flat, %c32_inv : index
    %zero = arith.constant 0 : index
    lego.yield %i_out, %zero : index, index
  } : !lego.layout

  %c0_5 = arith.constant 0 : index
  %i = arith.addi %tid, %c0_5 : index
  %j = arith.constant 0 : index
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----

// Test 8: RegP row-major conflict-free (non-GenP layout, should pass)
func.func @test_regp_conflict_free(%base_tid: index {lego.thread_id}) {
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index

  // RegP identity perm = row-major, 32 columns → each thread hits different bank
  %layout = lego.reg_p perm [0, 1] dims[%c4, %c32] : !lego.layout

  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----

// Test 9: RegP column-major bank conflict (non-GenP layout, should warn)
func.func @test_regp_col_conflict(%base_tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c4 = arith.constant 4 : index

  // RegP reversed perm = col-major: flat = j*32 + i
  // Threads access consecutive j → addresses stride by 32 → same bank
  %layout = lego.reg_p perm [1, 0] dims[%c32, %c4] : !lego.layout

  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0 : index
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----
