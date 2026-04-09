// RUN: lego-opt -lego-verify-coalescing %s -split-input-file -verify-diagnostics

// Test 1: Row-major layout with consecutive thread indexing (should pass - coalesced)
func.func @test_row_major_coalesced(%base_tid: index {lego.thread_id}) {
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index

  // Row-major: flat = i * 32 + j
  // When threads access consecutive elements (tid maps to j), this is coalesced
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

  // Simulate warp access: threadIdx → (0, threadIdx)
  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----

// Test 2: Column-major layout with consecutive thread indexing (should warn - NOT coalesced)
func.func @test_col_major_not_coalesced(%base_tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c4 = arith.constant 4 : index

  // Column-major: flat = j * 32 + i
  // When threads access consecutive j values, addresses jump by stride=32 (not unit stride)
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

  // Threads access consecutive j indices (stride-32 access)
  %i = arith.constant 0 : index
  %c0_1 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0_1 : index
  // expected-warning@+1 {{Layout may produce non-coalesced memory accesses}}
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----

// Test 3: Strided access pattern (should warn)
func.func @test_strided_not_coalesced(%base_tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index

  // Layout with stride-2 access
  %layout = lego.gen_p [%c32, %c64] apply (%i: index, %j: index) {
    %c2 = arith.constant 2 : index
    %c64_apply = arith.constant 64 : index

    // Multiply j by 2 (creates stride-2 pattern)
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
  %c0_2 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0_2 : index
  // expected-warning@+1 {{Layout may produce non-coalesced memory accesses}}
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----

// Test 4: Identity layout (1D) - should pass
func.func @test_identity_coalesced(%tid: index {lego.thread_id}) {
  %c1024 = arith.constant 1024 : index

  // 1D identity: flat = i
  %layout = lego.gen_p [%c1024] apply (%i: index) {
    lego.yield %i : index
  } inv (%flat: index) {
    lego.yield %flat : index
  } : !lego.layout

  %c0_3 = arith.constant 0 : index
  %i = arith.addi %tid, %c0_3 : index
  %addr = lego.apply %layout(%i) : !lego.layout
  return
}

// -----

// Test 5: RegP row-major (should pass - coalesced, non-GenP layout)
func.func @test_regp_row_coalesced(%base_tid: index {lego.thread_id}) {
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index

  // RegP with identity perm [0,1] = row-major: flat = i*32 + j
  %layout = lego.reg_p perm [0, 1] dims[%c4, %c32] : !lego.layout

  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----

// Test 6: RegP column-major (should warn - NOT coalesced, non-GenP layout)
func.func @test_regp_col_not_coalesced(%base_tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c4 = arith.constant 4 : index

  // RegP with reversed perm [1,0] = col-major: flat = j*32 + i
  %layout = lego.reg_p perm [1, 0] dims[%c32, %c4] : !lego.layout

  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0 : index
  // expected-warning@+1 {{Layout may produce non-coalesced memory accesses}}
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----

// Test 7: OrderBy with RegP inner (should pass - coalesced)
func.func @test_orderby_regp_coalesced(%base_tid: index {lego.thread_id}) {
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index

  %inner = lego.reg_p perm [0, 1] dims[%c4, %c32] : !lego.layout
  %layout = lego.order_by(%inner) : !lego.layout

  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %base_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}
