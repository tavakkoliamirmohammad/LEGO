// RUN: lego-opt %s -lego-desugar -lego-to-arith -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_basic_view
// CHECK-SAME: (%[[MEM:.*]]: memref<100xf32>, %[[I:.*]]: index, %[[J:.*]]: index, %[[VAL:.*]]: f32)
func.func @test_basic_view(%mem: memref<100xf32>, %i: index, %j: index, %val: f32) -> f32 {
  // Define layout: Row(10, 10)
  %row = lego.row [10, 10] : !lego.layout
  %ob = lego.order_by (%row) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[10, 10]] : !lego.layout

  // Create View
  %view = lego.cast_view %mem, %tiled : memref<100xf32>, !lego.layout -> !lego.view<f32>
  
  // Store logic: view[i, j] = val
  // CHECK: %[[C10:.*]] = arith.constant 10 : index
  // CHECK: %[[MUL:.*]] = arith.muli %[[I]], %[[C10]] : index
  // CHECK: %[[FLAT:.*]] = arith.addi {{.*}} : index
  // CHECK: memref.store %[[VAL]], %[[MEM]][%[[FLAT]]]
  lego.store %val, %view[%i, %j] : f32, !lego.view<f32>, index, index
  
  // Load logic: ret = view[i, j]
  // CHECK: %[[FLAT_LOAD:.*]] = arith.addi {{.*}} : index
  // CHECK: %[[RET:.*]] = memref.load %[[MEM]][%[[FLAT_LOAD]]]
  %ret = lego.load %view[%i, %j] : !lego.view<f32>, index, index
  return %ret : f32
}

// -----

// CHECK-LABEL: func @test_tiled_view
// CHECK-SAME: (%[[MEM:.*]]: memref<1024xf32>, %[[BI:.*]]: index, %[[BJ:.*]]: index, %[[LI:.*]]: index, %[[LJ:.*]]: index)
func.func @test_tiled_view(%mem: memref<1024xf32>, %bi: index, %bj: index, %li: index, %lj: index) -> f32 {
  // Layout: OrderBy(Row(32, 32)).TileBy([[8, 8], [4, 4]])
  %row = lego.row [32, 32] : !lego.layout
  %ob = lego.order_by(%row) : !lego.layout
  %tiled = lego.tile_by %ob tile_dims [[8, 8], [4, 4]] : !lego.layout
  
  %view = lego.cast_view %mem, %tiled : memref<1024xf32>, !lego.layout -> !lego.view<f32>
  
  // CHECK: arith.muli
  // CHECK: arith.addi
  // CHECK: %[[VAL:.*]] = memref.load %[[MEM]][%{{.*}}]
  // CHECK: return %[[VAL]]
  %val = lego.load %view[%bi, %bj, %li, %lj] : !lego.view<f32>, index, index, index, index
  return %val : f32
}

// -----

// CHECK-LABEL: func @test_col_major_view
// CHECK-SAME: (%[[MEM:.*]]: memref<100xf32>, %[[I:.*]]: index, %[[J:.*]]: index)
func.func @test_col_major_view(%mem: memref<100xf32>, %i: index, %j: index) -> f32 {
  %col = lego.col [10, 10] : !lego.layout
  %view = lego.cast_view %mem, %col : memref<100xf32>, !lego.layout -> !lego.view<f32>
  
  // CHECK: %[[C10:.*]] = arith.constant 10 : index
  // CHECK: %[[MUL:.*]] = arith.muli %[[J]], %[[C10]] : index
  // CHECK: %[[FLAT:.*]] = arith.addi {{.*}} : index
  // CHECK: %[[VAL:.*]] = memref.load %[[MEM]][%[[FLAT]]]
  %val = lego.load %view[%i, %j] : !lego.view<f32>, index, index
  return %val : f32
}

// -----

// CHECK-LABEL: func @test_gen_p_view
// CHECK-SAME: (%[[MEM:.*]]: memref<16xf32>, %[[I:.*]]: index, %[[J:.*]]: index)
func.func @test_gen_p_view(%mem: memref<16xf32>, %i: index, %j: index) -> f32 {
  %layout = lego.gen_p [4, 4]
    apply (%a: index, %b: index) {
      %sum = arith.addi %a, %b : index
      lego.yield %sum : index
    }
    inv (%flat: index) { 
        %c = arith.constant 0 : index
        lego.yield %c, %c : index, index
    } : !lego.layout
    
  %view = lego.cast_view %mem, %layout : memref<16xf32>, !lego.layout -> !lego.view<f32>
  
  // CHECK: %[[FLAT:.*]] = arith.addi {{.*}} : index
  // CHECK: %[[RET:.*]] = memref.load %[[MEM]][%[[FLAT]]]
  %val = lego.load %view[%i, %j] : !lego.view<f32>, index, index
  return %val : f32
}
