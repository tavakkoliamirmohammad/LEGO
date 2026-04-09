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
