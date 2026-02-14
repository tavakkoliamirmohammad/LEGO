// RUN: lego-opt %s -verify-diagnostics -split-input-file

// -----

func.func @test_tile_by_invalid_d() {
  %r = lego.row [10] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout // d=1, q=1
  // expected-error @+1 {{Inner tile dimension 2 does not match input layout dimension 1}}
  %tb = lego.tile_by %ob tile_dims [[2, 5]] : !lego.layout
  return
}

// -----

func.func @test_tile_by_product_mismatch() {
  %r = lego.row [10] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout // d=1, product=10
  // Inner list size d=1 matches.
  // Product 2*2 = 4 != 10
  // expected-error @+1 {{Total product of tile dims (4) does not match total product of input dims (10)}}
  %tb = lego.tile_by %ob tile_dims [[2], [2]] : !lego.layout
  return
}

// -----

func.func @test_regp_invalid_perm_rank() {
  // expected-error @+1 {{Permutation rank 2 does not match dimensions rank 3}}
  %p = lego.reg_p perm [1, 0] dims [10, 10, 10] : !lego.layout
  return
}

// -----

func.func @test_regp_invalid_perm_content() {
  // expected-error @+1 {{Invalid permutation: not a permutation of 0..2}}
  %p = lego.reg_p perm [0, 1, 3] dims [10, 10, 10] : !lego.layout
  return
}
