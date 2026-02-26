// RUN: lego-opt -lego-verify-coalescing='warp-size=16' %s -split-input-file -verify-diagnostics

// Test 1: Custom pass options (should pass)
// With a 16-thread warp, a layout that accesses 16 consecutive words is coalesced.
func.func @test_custom_options(%custom_tid: index {lego.thread_id}) {
  %c16 = arith.constant 16 : index
  %c4 = arith.constant 4 : index

  // Row-major: flat = i * 16 + j
  // When threads access consecutive elements (tid maps to j), this is coalesced
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

  // Simulate warp access: threadIdx → (0, threadIdx)
  %i = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %j = arith.addi %custom_tid, %c0 : index
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}

// -----

// Test 2: Custom pass options (should warn - NOT coalesced)
func.func @test_custom_options_uncoalesced(%custom_tid: index {lego.thread_id}) {
  %c16 = arith.constant 16 : index
  %c4 = arith.constant 4 : index

  // Column-major: flat = j * 16 + i
  // When threads access consecutive j values, addresses jump by stride=16 (not unit stride)
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

  // Threads access consecutive j indices (stride-16 access)
  %i = arith.constant 0 : index
  %c0_1 = arith.constant 0 : index
  %j = arith.addi %custom_tid, %c0_1 : index
  // expected-warning@+1 {{Layout may produce non-coalesced memory accesses}}
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}
