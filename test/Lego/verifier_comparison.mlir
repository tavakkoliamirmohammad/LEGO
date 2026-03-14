// RUN: lego-opt %s -lego-generate-bounds-checks --lego-normalization --lego-to-arith --canonicalize --cse --lego-arith-simplification --int-range-optimizations --canonicalize --cse -lego-external-smt-verifier -split-input-file -verify-diagnostics

// ============================================================================
// gen_p: Symbolic vs Concrete
// ============================================================================

func.func @gen_p_concrete_pass(%i: index, %j: index) -> index {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  lego.assume_bounds %i lb : %c0 ub : %c10
  lego.assume_bounds %j lb : %c0 ub : %c10

  %layout = lego.gen_p [%c10, %c10]
    apply (%arg0: index, %arg1: index) {
      %mul = arith.muli %arg0, %c10 : index
      %res = arith.addi %mul, %arg1 : index
      lego.yield %res : index
    }
    inv (%flat: index) {
      %ii = arith.divui %flat, %c10 : index
      %jj = arith.remui %flat, %c10 : index
      lego.yield %ii, %jj : index, index
    } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// -----
func.func @gen_p_symbolic_pass(%i: index, %j: index, %N: index, %M: index) -> index {
  %c0 = arith.constant 0 : index
  lego.assume_bounds %i lb : %c0 ub : %N
  lego.assume_bounds %j lb : %c0 ub : %M

  %layout = lego.gen_p [%N, %M]
    apply (%arg0: index, %arg1: index) {
      %mul = arith.muli %arg0, %M : index
      %res = arith.addi %mul, %arg1 : index
      lego.yield %res : index
    }
    inv (%flat: index) {
      %ii = arith.divui %flat, %M : index
      %jj = arith.remui %flat, %M : index
      lego.yield %ii, %jj : index, index
    } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// -----
func.func @gen_p_concrete_fail(%i: index, %j: index) -> index {
  %c10 = arith.constant 10 : index
  %c100 = arith.constant 100 : index
  %c0 = arith.constant 0 : index
  lego.assume_bounds %i lb : %c0 ub : %c100 // Possible i up to 99
  lego.assume_bounds %j lb : %c0 ub : %c10

  %layout = lego.gen_p [%c10, %c10]
    apply (%arg0: index, %arg1: index) {
      %mul = arith.muli %arg0, %c10 : index
      %res = arith.addi %mul, %arg1 : index
      lego.yield %res : index
    }
    inv (%flat: index) {
      %ii = arith.divui %flat, %c10 : index
      %jj = arith.remui %flat, %c10 : index
      lego.yield %ii, %jj : index, index
    } : !lego.layout
  // expected-error@+1 {{Out-of-bounds access is possible (proven by Z3)}}
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// -----
func.func @gen_p_symbolic_fail(%i: index, %j: index, %N: index, %M: index, %K: index) -> index {
  %c0 = arith.constant 0 : index
  lego.assume_bounds %i lb : %c0 ub : %K // K could be > N
  lego.assume_bounds %j lb : %c0 ub : %M

  %layout = lego.gen_p [%N, %M]
    apply (%arg0: index, %arg1: index) {
      %mul = arith.muli %arg0, %M : index
      %res = arith.addi %mul, %arg1 : index
      lego.yield %res : index
    }
    inv (%flat: index) {
      %ii = arith.divui %flat, %M : index
      %jj = arith.remui %flat, %M : index
      lego.yield %ii, %jj : index, index
    } : !lego.layout
  // expected-error@+1 {{Out-of-bounds access is possible (proven by Z3)}}
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// ============================================================================
// tile_by: Symbolic vs Concrete
// ============================================================================

// -----
func.func @tile_by_concrete_pass(%i: index, %j: index) -> index {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  lego.assume_bounds %i lb : %c0 ub : %c10
  lego.assume_bounds %j lb : %c0 ub : %c10

  %r = lego.row [%c10, %c10] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout
  // Matching input volume and rank
  %tb = lego.tile_by %ob tile_dims [[%c10, %c10]] : !lego.layout
  %f = lego.apply %tb(%i, %j) : !lego.layout
  return %f : index
}

// -----
func.func @tile_by_symbolic_pass(%i: index, %j: index, %N: index, %M: index) -> index {
  %c0 = arith.constant 0 : index
  lego.assume_bounds %i lb : %c0 ub : %N
  lego.assume_bounds %j lb : %c0 ub : %M

  %r = lego.row [%N, %M] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout
  %tb = lego.tile_by %ob tile_dims [[%N, %M]] : !lego.layout
  %f = lego.apply %tb(%i, %j) : !lego.layout
  return %f : index
}

// -----
func.func @tile_by_symbolic_bounded_pass(%i: index, %j: index, %N: index, %M: index) -> index {
  %c0 = arith.constant 0 : index
  // Explicitly assume i < N and j < M
  lego.assume_bounds %i lb : %c0 ub : %N
  lego.assume_bounds %j lb : %c0 ub : %M

  %r = lego.row [%N, %M] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout
  %tb = lego.tile_by %ob tile_dims [[%N, %M]] : !lego.layout
  %f = lego.apply %tb(%i, %j) : !lego.layout
  return %f : index
}

// -----
func.func @tile_by_symbolic_bounded_fail(%i: index, %j: index, %N: index, %M: index, %K: index) -> index {
  %c0 = arith.constant 0 : index
  // i is bounded by K, which is unknown relative to N
  lego.assume_bounds %i lb : %c0 ub : %K
  lego.assume_bounds %j lb : %c0 ub : %M

  %r = lego.row [%N, %M] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout
  %tb = lego.tile_by %ob tile_dims [[%N, %M]] : !lego.layout
  // expected-error@+1 {{Out-of-bounds access is possible (proven by Z3)}}
  %f = lego.apply %tb(%i, %j) : !lego.layout
  return %f : index
}

// ============================================================================
// 4D Tiling: (2x2) x (5x5)
// ============================================================================

// -----
func.func @tile_by_4d_pass(%i0: index, %i1: index, %i2: index, %i3: index) -> index {
  %c10 = arith.constant 10 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %c0 = arith.constant 0 : index
  
  lego.assume_bounds %i0 lb : %c0 ub : %c2
  lego.assume_bounds %i1 lb : %c0 ub : %c2
  lego.assume_bounds %i2 lb : %c0 ub : %c5
  lego.assume_bounds %i3 lb : %c0 ub : %c5

  %r = lego.row [%c10, %c10] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout
  // 4D tiled layout: [2, 2, 5, 5]
  %tb = lego.tile_by %ob tile_dims [[%c2, %c2], [%c5, %c5]] : !lego.layout
  %f = lego.apply %tb(%i0, %i1, %i2, %i3) : !lego.layout
  return %f : index
}

// -----
func.func @tile_by_4d_fail(%i0: index, %i1: index, %i2: index, %i3: index) -> index {
  %c10 = arith.constant 10 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %c100 = arith.constant 100 : index
  %c0 = arith.constant 0 : index
  
  lego.assume_bounds %i0 lb : %c0 ub : %c2
  lego.assume_bounds %i1 lb : %c0 ub : %c2
  lego.assume_bounds %i2 lb : %c0 ub : %c100 // Possible i2 up to 99, but limit is 5
  lego.assume_bounds %i3 lb : %c0 ub : %c5

  %r = lego.row [%c10, %c10] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout
  %tb = lego.tile_by %ob tile_dims [[%c2, %c2], [%c5, %c5]] : !lego.layout
  // expected-error@+1 {{Out-of-bounds access is possible (proven by Z3)}}
  %f = lego.apply %tb(%i0, %i1, %i2, %i3) : !lego.layout
  return %f : index
}
