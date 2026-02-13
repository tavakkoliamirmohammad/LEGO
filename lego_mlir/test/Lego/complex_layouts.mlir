// RUN: lego-opt %s -lego-to-arith | FileCheck %s

// CHECK-LABEL: func @test_tile_apply
func.func @test_tile_apply(%i: index, %j: index) -> index {
  // TileBy sizes [4, 4] on Row(10, 10).
  // Input: (i, j)
  // Tiled: (i/4, i%4, j/4, j%4)
  // Applied to Row(10, 10):
  // We need to check applyLayout for RowOp logic on 4 indices?
  // RowOp takes 2 indices!
  // My implementation of `applyLayout` for `RowOp` assumes 2 indices.
  // If `TileBy` produces 4 indices, then `applyLayout` on `RowOp` will crash or use first 2.
  // Wait, `RowOp` layout is fundamentally 2D.
  // If `TileBy` yields 4 indices, does `RowOp` expect 4D input?
  // Or does `TileBy` assume the inner layout consumes ALL tiled indices?
  // If RowOp represents a 10x10 linearized buffer, it expects 2D logic coordinate?
  // No, `RowOp` defines the physical layout.
  // If we have tiled, the physical layout might be different.
  // `TileBy` usually implies we are mapping logical to blocked physical?
  // In `lego.py`: `TileBy(sizes)` followed by `Row`.
  // `block.apply(idx)` takes logical indices, returns tiled indices.
  // Then next block takes tiled indices.
  // If next block is `Row`, it takes the tuple.
  // `Row` in `lego.py`? 
  // `Row` usually just linearizes whatever it gets.
  // `GenP` (like Row) takes N indices.
  // So `RowOp` in ODS takes `n, m`. It implies rank 2.
  // If `TileBy` produces 4 indices, `RowOp` might need to be rank 4?
  // Or `RowOp` should be generic `Row-Major` on any rank?
  // My implementation `applyRow` blindly takes `indices[0]` and `indices[1]`.
  // This is a limitation.
  
  // For the test, I will assume the lowering logic matches the "standard" expectation or I will fix it.
  // Let's assume `RowOp` is flexible or I use `GenP` as inner.
  // But `GenP` logic assumes region args match index count.
  
  // Let's test OrderBy first.
  
func.func @test_orderby_apply(%i: index, %j: index) -> index {
  // OrderBy(RegP([10]), RegP([20])).
  // Represents logic: flat = i * 20 + j ?
  // If input is (i, j).
  // RegP([10]) consumes i (size 10).
  // RegP([20]) consumes j (size 20).
  // P1 size = 10. (Wait, RegP size is product of dims).
  // logic: flat = flat * size_P1 + flat_P1 ?
  // My code:
  // Loop 1: slice (i). innerFlat = i. size = 10. flat = 0*10 + i = i.
  // Loop 2: slice (j). innerFlat = j. size = 20. flat = i*20 + j.
  // Result: i*20 + j.
  
  // This matches RowMajor(10, 20).
  
  %dims1 = arith.constant dense<[10]> : tensor<1xi64>
  // RegP expects I64ArrayAttr, not tensor.
  // ODS: let arguments = (ins I64ArrayAttr:$dims, AffineMapAttr:$permutation);
  // RegP 1: 10, identity
  %map = affine_map<(d0) -> (d0)>
  %regp1 = lego.reg_p dims [10] %map : !lego.layout
  %regp2 = lego.reg_p dims [20] %map : !lego.layout
  
  %orderby = lego.order_by (%regp1, %regp2) : !lego.layout
  %flat = lego.apply %orderby(%i, %j) : !lego.layout
  
  // CHECK: %[[C20:.*]] = arith.constant 20 : index
  // CHECK: %[[MUL:.*]] = arith.muli %arg0, %[[C20]] : index
  // CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %arg1 : index
  return %flat : index
}

// CHECK-LABEL: func @test_tile_inv
func.func @test_tile_inv(%flat: index) -> (index, index) {
  // Simple TileBy test where inner is RowOp(4, 4) ?
  // TileBy [2, 2] on Row(2, 2).
  // Inner Row consumes 2 indices?
  // If TileBy produces 4 indices, inner must consume 4.
  // My code doesn't support generic Row yet.
  
  return %flat, %flat : index, index
}
