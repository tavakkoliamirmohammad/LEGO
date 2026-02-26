// RUN: lego-opt -lego-verify-bank-conflicts='warp-size=8 num-banks=16 element-size=2' %s -split-input-file -verify-diagnostics

// Test 1: Custom pass options (should pass)
func.func @test_custom_options(%custom_tid: index {lego.thread_id}) {
  %c16 = arith.constant 16 : index
  %c4 = arith.constant 4 : index

  // With a 16-bank memory and 2-byte elements, bank = (addr * 2 / 4) % 16
  // Threads access addr = tid * 4 (stride 4 in elements = stride 8 in bytes)
  // bank_t = (t * 8 / 4) % 16 = (2 * t) % 16
  // For t in 0..7, bank_t is in {0, 2, 4, 6, 8, 10, 12, 14}. No conflicts.
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

  return
}

// -----

// Test 2: Custom pass options (should fail because stride causes conflicts in 16-bank memory)
func.func @test_custom_options_conflict(%custom_tid: index {lego.thread_id}) {
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index

  // Threads access addr = tid * 8 (stride 8 in elements = stride 16 in bytes)
  // With 16 banks and 2-byte elements: bank_t = (t * 16 / 4) % 16 = (4 * t) % 16
  // Thread 0 maps to bank 0 (addr 0). Thread 4 maps to bank 0 (addr 32). Conflict!
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
  // expected-warning@+1 {{Layout may cause shared memory bank conflicts}}
  %addr = lego.apply %layout(%i, %j) : !lego.layout

  return
}
