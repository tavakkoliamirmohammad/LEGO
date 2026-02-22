// RUN: lego-opt %s -lego-generate-bounds-checks -lego-lower -lego-external-smt-verifier -split-input-file -verify-diagnostics

// -----

func.func @test_bounds_pass(%N: index, %i: index, %j: index) -> index {
  // We assume N is at most 10
  %c10 = arith.constant 10 : index
  %c5 = arith.constant 5 : index
  %c0 = arith.constant 0 : index

  lego.assume_bounds %N ub : %c10
  lego.assume_bounds %i lb : %c0 ub : %N
  lego.assume_bounds %j lb : %c0 ub : %c5

  // Layout expects dims: [10, 5]
  %layout = lego.row [%c10, %c5] : !lego.layout
  
  // Verifier checks that `i >= 0 && i < 10` and `j >= 0 && j < 5`.
  %flat = lego.apply %layout(%i, %j) : !lego.layout
  return %flat : index
}

// -----

func.func @test_bounds_fail(%i: index, %j: index) {
  %c100 = arith.constant 100 : index
  %c5 = arith.constant 5 : index
  %c0 = arith.constant 0 : index

  lego.assume_bounds %i lb : %c0 ub : %c100
  lego.assume_bounds %j lb : %c0 ub : %c5

  // Layout expects dims: [10, 5]
  %c10 = arith.constant 10 : index
  %layout = lego.row [%c10, %c5] : !lego.layout
  
  // Expected to fail because i can be up to 99, but layout expects i < 10.
  // expected-error@+1 {{Out-of-bounds access is possible (proven by Z3)}}
  %flat = lego.apply %layout(%i, %j) : !lego.layout
  return
}

// -----

func.func @test_inv_bounds_fail(%flat: index) {
  %c1000 = arith.constant 1000 : index
  lego.assume_bounds %flat ub : %c1000

  %c10 = arith.constant 10 : index
  %c5 = arith.constant 5 : index
  %layout = lego.col [%c10, %c5] : !lego.layout
  
  // Expected to fail because layout volume is 50, but flat can be 999.
  // expected-error@+1 {{Out-of-bounds flat index is possible (proven by Z3)}}
  %idx:2 = lego.apply_inverse %layout(%flat) : !lego.layout -> index, index
  return
}

// -----
func.func @test_inv_bounds_pass(%flat: index) {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  lego.assume_bounds %flat lb : %c0 ub : %c10

  %c5 = arith.constant 5 : index
  %layout = lego.col [%c10, %c5] : !lego.layout
  
  // Expected to pass since flat < 10 and volume is 50.
  %idx:2 = lego.apply_inverse %layout(%flat) : !lego.layout -> index, index
  return
}
