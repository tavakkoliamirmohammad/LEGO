// RUN: lego-opt %s | FileCheck %s

// CHECK-LABEL: func @test_check_coalescing
func.func @test_check_coalescing(%tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c4 = arith.constant 4 : index
  %layout = lego.reg_p perm [0, 1] dims[%c4, %c32] : !lego.layout
  %c0 = arith.constant 0 : index
  %j = arith.addi %tid, %c0 : index
  %addr = lego.apply %layout(%c0, %j) : !lego.layout
  // CHECK: lego.check %{{.*}} {coalescing}
  lego.check %addr {coalescing}
  return
}

// CHECK-LABEL: func @test_check_bank_conflicts
func.func @test_check_bank_conflicts(%tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %c4 = arith.constant 4 : index
  %layout = lego.reg_p perm [0, 1] dims[%c4, %c32] : !lego.layout
  %c0 = arith.constant 0 : index
  %j = arith.addi %tid, %c0 : index
  %addr = lego.apply %layout(%c0, %j) : !lego.layout
  // CHECK: lego.check %{{.*}} {bank_conflict_free, num_banks = 16 : i64, warp_size = 16 : i64}
  lego.check %addr {bank_conflict_free, warp_size = 16 : i64, num_banks = 16 : i64}
  return
}

// CHECK-LABEL: func @test_check_both
func.func @test_check_both(%tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %layout = lego.reg_p perm [0] dims[%c32] : !lego.layout
  %c0 = arith.constant 0 : index
  %idx = arith.addi %tid, %c0 : index
  %addr = lego.apply %layout(%idx) : !lego.layout
  // CHECK: lego.check %{{.*}} {bank_conflict_free, coalescing}
  lego.check %addr {coalescing, bank_conflict_free}
  return
}
