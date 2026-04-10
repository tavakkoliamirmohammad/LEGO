// RUN: lego-opt -lego-lower %s -split-input-file -verify-diagnostics

// Test 1: Row-major RegP — coalesced (should pass)
func.func @test_regp_row_coalesced(%tid: index {lego.thread_id}) {
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %layout = lego.reg_p perm [0, 1] dims[%c4, %c32] : !lego.layout
  %c0 = arith.constant 0 : index
  %j = arith.addi %tid, %c0 : index
  %addr = lego.apply %layout(%c0, %j) : !lego.layout
  lego.check %addr {coalescing}
  return
}

// -----

// Test 2: Column-major RegP — NOT coalesced (should warn)
func.func @test_regp_col_not_coalesced(%tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c4 = arith.constant 4 : index
  %layout = lego.reg_p perm [1, 0] dims[%c32, %c4] : !lego.layout
  %c0 = arith.constant 0 : index
  %j = arith.addi %tid, %c0 : index
  %addr = lego.apply %layout(%c0, %j) : !lego.layout
  // expected-warning@+1 {{Layout may produce non-coalesced memory accesses}}
  lego.check %addr {coalescing}
  return
}

// -----

// Test 3: Row-major GenP — coalesced (should pass)
func.func @test_genp_row_coalesced(%tid: index {lego.thread_id}) {
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
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
  %c0 = arith.constant 0 : index
  %j = arith.addi %tid, %c0 : index
  %addr = lego.apply %layout(%c0, %j) : !lego.layout
  lego.check %addr {coalescing}
  return
}

// -----

// Test 4: Row-major RegP — bank-conflict free (should pass)
func.func @test_regp_row_no_conflict(%tid: index {lego.thread_id}) {
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %layout = lego.reg_p perm [0, 1] dims[%c4, %c32] : !lego.layout
  %c0 = arith.constant 0 : index
  %j = arith.addi %tid, %c0 : index
  %addr = lego.apply %layout(%c0, %j) : !lego.layout
  lego.check %addr {bank_conflict_free}
  return
}

// -----

// Test 5: Column-major RegP — bank conflicts (should warn)
func.func @test_regp_col_conflict(%tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c4 = arith.constant 4 : index
  %layout = lego.reg_p perm [1, 0] dims[%c32, %c4] : !lego.layout
  %c0 = arith.constant 0 : index
  %j = arith.addi %tid, %c0 : index
  %addr = lego.apply %layout(%c0, %j) : !lego.layout
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  lego.check %addr {bank_conflict_free}
  return
}

// -----

// Test 6: Both checks on same address (row-major, should pass both)
func.func @test_both_checks(%tid: index {lego.thread_id}) {
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %layout = lego.reg_p perm [0, 1] dims[%c4, %c32] : !lego.layout
  %c0 = arith.constant 0 : index
  %j = arith.addi %tid, %c0 : index
  %addr = lego.apply %layout(%c0, %j) : !lego.layout
  lego.check %addr {coalescing, bank_conflict_free}
  return
}

// -----

// Test 7: Custom warp size (16-thread warp, row-major, should pass)
func.func @test_custom_warp_size(%tid: index {lego.thread_id}) {
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %layout = lego.reg_p perm [0, 1] dims[%c4, %c16] : !lego.layout
  %c0 = arith.constant 0 : index
  %j = arith.addi %tid, %c0 : index
  %addr = lego.apply %layout(%c0, %j) : !lego.layout
  lego.check %addr {coalescing, warp_size = 16 : i64}
  return
}

// -----

// Test 8: OrderBy with RegP inner — coalesced (should pass)
func.func @test_orderby_coalesced(%tid: index {lego.thread_id}) {
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %inner = lego.reg_p perm [0, 1] dims[%c4, %c32] : !lego.layout
  %layout = lego.order_by(%inner) : !lego.layout
  %c0 = arith.constant 0 : index
  %j = arith.addi %tid, %c0 : index
  %addr = lego.apply %layout(%c0, %j) : !lego.layout
  lego.check %addr {coalescing}
  return
}

// -----

// Test 9: TileBy layout — coalesced (should pass)
func.func @test_tileby_coalesced(%tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %row = lego.row [%c32, %c32] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[%c32, %c32]] : !lego.layout
  %c0 = arith.constant 0 : index
  %j = arith.addi %tid, %c0 : index
  %addr = lego.apply %tiled(%c0, %j) : !lego.layout
  lego.check %addr {coalescing}
  return
}

// -----

// Test 10: Column-major GenP — bank conflict (should warn)
func.func @test_genp_col_bank_conflict(%base_tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c4 = arith.constant 4 : index

  // Column-major: flat = j * 32 + i
  // When threads access consecutive j, all addresses map to same bank
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

  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  lego.check %addr {bank_conflict_free}
  return
}

// -----

// Test 11: 2-way bank conflict GenP (should warn)
func.func @test_2way_bank_conflict(%base_tid: index {lego.thread_id}) {
  %c16 = arith.constant 16 : index
  %c2 = arith.constant 2 : index

  // Each pair of consecutive i values maps to same bank
  %layout = lego.gen_p [%c16, %c2] apply (%i: index, %j: index) {
    %c2_apply = arith.constant 2 : index
    %c16_apply = arith.constant 16 : index
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

  %c0 = arith.constant 0 : index
  %i = arith.addi %base_tid, %c0 : index
  %j = arith.constant 0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  lego.check %addr {bank_conflict_free}
  return
}

// -----

// Test 12: Padded layout to avoid conflicts (should pass)
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

  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout
  lego.check %addr {bank_conflict_free}
  return
}

// -----

// Test 13: Strided access causing bank conflicts (should warn)
func.func @test_strided_bank_conflicts(%base_tid: index {lego.thread_id}) {
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index

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
  %c2 = arith.constant 2 : index
  %i_scaled = arith.muli %base_tid, %c2 : index
  %j = arith.constant 0 : index
  %addr = lego.apply %layout(%i_scaled, %j) : !lego.layout
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  lego.check %addr {bank_conflict_free}
  return
}

// -----

// Test 14: Transpose pattern bank conflicts (should warn)
func.func @test_transpose_bank_conflicts(%base_tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index

  // Transpose: swap i and j → flat = j * 32 + i
  %layout = lego.gen_p [%c32, %c32] apply (%i: index, %j: index) {
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

  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  lego.check %addr {bank_conflict_free}
  return
}

// -----

// Test 15: Worst-case bank conflict — all threads same bank (should warn)
func.func @test_worst_case_bank_conflict(%tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index

  // flat = tid * 32: addresses 0, 32, 64, 96, ... all map to bank 0
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

  %c0 = arith.constant 0 : index
  %i = arith.addi %tid, %c0 : index
  %j = arith.constant 0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  lego.check %addr {bank_conflict_free}
  return
}

// -----

// Test 16: Row-major GenP conflict-free bank access (should pass)
func.func @test_genp_row_no_bank_conflict(%base_tid: index {lego.thread_id}) {
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index

  // Row-major 4x32: each thread in a warp accesses a different column/bank
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

  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout
  lego.check %addr {bank_conflict_free}
  return
}

// -----

// Test 17: Custom bank options — conflict-free (warp=8, banks=16, elem=2, should pass)
func.func @test_custom_bank_options_pass(%custom_tid: index {lego.thread_id}) {
  %c16 = arith.constant 16 : index
  %c4 = arith.constant 4 : index

  // With 16 banks and 2-byte elements, bank = (addr * 2 / 4) % 16
  // Threads access addr = tid * 4 (stride 4 in elements = stride 8 in bytes)
  // bank_t = (t * 8 / 4) % 16 = (2 * t) % 16
  // For t in 0..7, banks are {0, 2, 4, 6, 8, 10, 12, 14}. No conflicts.
  %layout = lego.gen_p [%c4, %c16] apply (%i: index, %j: index) {
    %c16_apply = arith.constant 16 : index
    %t = arith.muli %i, %c16_apply : index
    %flat = arith.addi %t, %j : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c16_inv = arith.constant 16 : index
    %i_out = arith.divui %flat, %c16_inv : index
    %j_out = arith.remui %flat, %c16_inv : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  %c4_stride = arith.constant 4 : index
  %j = arith.muli %custom_tid, %c4_stride : index
  %i = arith.constant 0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout
  lego.check %addr {bank_conflict_free, warp_size = 8 : i64, num_banks = 16 : i64, element_size = 2 : i64}
  return
}

// -----

// Test 18: Custom bank options — conflict (warp=8, banks=16, elem=2, should warn)
func.func @test_custom_bank_options_conflict(%custom_tid: index {lego.thread_id}) {
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index

  // Threads access addr = tid * 8 (stride 8 in elements = stride 16 in bytes)
  // With 16 banks and 2-byte elements: bank_t = (t * 16 / 4) % 16 = (4 * t) % 16
  // Thread 0 → bank 0, Thread 4 → bank 0. Conflict!
  %layout = lego.gen_p [%c8, %c16] apply (%i: index, %j: index) {
    %c16_apply = arith.constant 16 : index
    %t = arith.muli %i, %c16_apply : index
    %flat = arith.addi %t, %j : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c16_inv = arith.constant 16 : index
    %i_out = arith.divui %flat, %c16_inv : index
    %j_out = arith.remui %flat, %c16_inv : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  %c8_stride = arith.constant 8 : index
  %j = arith.muli %custom_tid, %c8_stride : index
  %i = arith.constant 0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  lego.check %addr {bank_conflict_free, warp_size = 8 : i64, num_banks = 16 : i64, element_size = 2 : i64}
  return
}

// -----

// Test 19: Column-major GenP — NOT coalesced (should warn)
func.func @test_genp_col_not_coalesced(%base_tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c4 = arith.constant 4 : index

  // Column-major: flat = j * 32 + i
  // Threads access consecutive j → addresses jump by stride=32
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

  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout
  // expected-warning@+1 {{Layout may produce non-coalesced memory accesses}}
  lego.check %addr {coalescing}
  return
}

// -----

// Test 20: Strided GenP — NOT coalesced (should warn)
func.func @test_genp_strided_not_coalesced(%base_tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index

  // Stride-2 pattern: j is multiplied by 2
  %layout = lego.gen_p [%c32, %c64] apply (%i: index, %j: index) {
    %c2 = arith.constant 2 : index
    %c64_apply = arith.constant 64 : index
    %j_scaled = arith.muli %j, %c2 : index
    %t = arith.muli %i, %c64_apply : index
    %flat = arith.addi %t, %j_scaled : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c64_inv = arith.constant 64 : index
    %c2 = arith.constant 2 : index
    %i_out = arith.divui %flat, %c64_inv : index
    %j_temp = arith.remui %flat, %c64_inv : index
    %j_out = arith.divui %j_temp, %c2 : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout
  // expected-warning@+1 {{Layout may produce non-coalesced memory accesses}}
  lego.check %addr {coalescing}
  return
}

// -----

// Test 21: 1D identity GenP — coalesced (should pass)
func.func @test_genp_1d_identity_coalesced(%tid: index {lego.thread_id}) {
  %c1024 = arith.constant 1024 : index

  // 1D identity: flat = i
  %layout = lego.gen_p [%c1024] apply (%i: index) {
    lego.yield %i : index
  } inv (%flat: index) {
    lego.yield %flat : index
  } : !lego.layout

  %c0 = arith.constant 0 : index
  %i = arith.addi %tid, %c0 : index
  %addr = lego.apply %layout(%i) : !lego.layout
  lego.check %addr {coalescing}
  return
}

// -----

// Test 22: Custom warp-size 16, GenP row-major — coalesced (should pass)
func.func @test_custom_warp16_coalesced(%custom_tid: index {lego.thread_id}) {
  %c16 = arith.constant 16 : index
  %c4 = arith.constant 4 : index

  %layout = lego.gen_p [%c4, %c16] apply (%i: index, %j: index) {
    %c16_apply = arith.constant 16 : index
    %t = arith.muli %i, %c16_apply : index
    %flat = arith.addi %t, %j : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c16_inv = arith.constant 16 : index
    %i_out = arith.divui %flat, %c16_inv : index
    %j_out = arith.remui %flat, %c16_inv : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %custom_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout
  lego.check %addr {coalescing, warp_size = 16 : i64}
  return
}

// -----

// Test 23: Custom warp-size 16, GenP col-major — NOT coalesced (should warn)
func.func @test_custom_warp16_not_coalesced(%custom_tid: index {lego.thread_id}) {
  %c16 = arith.constant 16 : index
  %c4 = arith.constant 4 : index

  // Column-major: flat = j * 16 + i
  %layout = lego.gen_p [%c16, %c4] apply (%i: index, %j: index) {
    %c16_apply = arith.constant 16 : index
    %t = arith.muli %j, %c16_apply : index
    %flat = arith.addi %t, %i : index
    lego.yield %flat : index
  } inv (%flat: index) {
    %c16_inv = arith.constant 16 : index
    %j_out = arith.divui %flat, %c16_inv : index
    %i_out = arith.remui %flat, %c16_inv : index
    lego.yield %i_out, %j_out : index, index
  } : !lego.layout

  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %custom_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout
  // expected-warning@+1 {{Layout may produce non-coalesced memory accesses}}
  lego.check %addr {coalescing, warp_size = 16 : i64}
  return
}
