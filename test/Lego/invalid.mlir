// RUN: lego-opt %s -verify-diagnostics -split-input-file

// -----

func.func @test_tile_by_invalid_d() {
  %c10 = arith.constant 10 : index
  %r = lego.row [%c10] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout // d=1, q=1
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  // expected-error @+1 {{'lego.tile_by' op Inner tile dimension 2 does not match input layout dimension 1}}
  %tb = lego.tile_by %ob tile_dims [[%c2, %c5]] : !lego.layout
  return
}

// -----

func.func @test_tile_by_product_mismatch() {
  %c10 = arith.constant 10 : index
  %r = lego.row [%c10] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout // d=1, product=10
  // Inner list size d=1 matches.
  // Product 2*2 = 4 != 10
  %c2 = arith.constant 2 : index
  // expected-error @+1 {{'lego.tile_by' op Total product of tile dims (4) does not match total product of input dims (10)}}
  %tb = lego.tile_by %ob tile_dims [[%c2], [%c2]] : !lego.layout
  return
}

// -----

func.func @test_regp_invalid_perm_rank() {
  %c10 = arith.constant 10 : index
  // expected-error @+1 {{'lego.reg_p' op Permutation rank 2 does not match dimensions rank 3}}
  %p = lego.reg_p perm [1, 0] dims [%c10, %c10, %c10] : !lego.layout
  return
}

// -----

func.func @test_regp_invalid_perm_content() {
  %c10 = arith.constant 10 : index
  // expected-error @+1 {{'lego.reg_p' op Invalid permutation: not a permutation of 0..2}}
  %p = lego.reg_p perm [0, 1, 3] dims [%c10, %c10, %c10] : !lego.layout
  return
}

// -----

func.func @test_row_zero_dim() {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'lego.row' op Dimension 0 must be strictly positive}}
  %r = lego.row [%c10, %c0] : !lego.layout
  return
}

// -----

func.func @test_col_neg_dim() {
  %c10 = arith.constant 10 : index
  %cm5 = arith.constant -5 : index
  // expected-error @+1 {{'lego.col' op Dimension -5 must be strictly positive}}
  %c = lego.col [%c10, %cm5] : !lego.layout
  return
}

// -----

func.func @test_regp_zero_dim() {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'lego.reg_p' op Dimension 0 must be strictly positive}}
  %p = lego.reg_p perm [0, 1] dims [%c10, %c0] : !lego.layout
  return
}

// -----

func.func @test_tileby_zero_dim() {
  %c10 = arith.constant 10 : index
  %r = lego.row [%c10] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{'lego.tile_by' op Tile dimension 0 must be strictly positive}}
  %tb = lego.tile_by %ob tile_dims [[%c0]] : !lego.layout
  return
}
// -----

func.func @test_cast_view_mismatch(%mem: memref<50xf32>) {
  %c10 = arith.constant 10 : index
  %row = lego.row [%c10, %c10] : !lego.layout
  // expected-error @+1 {{'lego.cast_view' op MemRef total number of elements (50) does not match layout volume (100)}}
  %view = lego.cast_view %mem, %row : memref<50xf32>, !lego.layout -> !lego.view<f32>
  return
}

// -----

func.func @apply_rank_mismatch(%i: index) {
  %c10 = arith.constant 10 : index
  %r = lego.row [%c10, %c10] : !lego.layout
  // expected-error@+1 {{'lego.apply' op number of indices (1) does not match layout rank (2)}}
  %f = lego.apply %r(%i) : !lego.layout
  return
}

// -----

func.func @test_tile_by_requires_orderby() {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %regp = lego.reg_p perm [0, 1] dims [%c4, %c8] : !lego.layout
  %c2 = arith.constant 2 : index
  // expected-error @+1 {{'lego.tile_by' op input must be an OrderBy layout}}
  %tb = lego.tile_by %regp tile_dims [[%c2, %c2], [%c2, %c4]] : !lego.layout
  return
}
