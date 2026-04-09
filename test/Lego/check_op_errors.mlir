// RUN: lego-opt %s -split-input-file -verify-diagnostics

// Error: no property specified
func.func @test_no_property(%tid: index {lego.thread_id}) {
  %c32 = arith.constant 32 : index
  %layout = lego.reg_p perm [0] dims[%c32] : !lego.layout
  %c0 = arith.constant 0 : index
  %addr = lego.apply %layout(%c0) : !lego.layout
  // expected-error@+1 {{must request at least one property}}
  lego.check %addr {}
  return
}

// -----

// Error: no lego.thread_id argument
func.func @test_no_thread_id(%x: index) {
  %c32 = arith.constant 32 : index
  %layout = lego.reg_p perm [0] dims[%c32] : !lego.layout
  %addr = lego.apply %layout(%x) : !lego.layout
  // expected-error@+1 {{parent function must have a {lego.thread_id} argument}}
  lego.check %addr {coalescing}
  return
}
