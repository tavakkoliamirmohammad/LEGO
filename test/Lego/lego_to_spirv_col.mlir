// RUN: lego-opt %s --lego-to-spirv | FileCheck %s

// Test: Column-major layout inside gpu.launch lowers to SPIR-V.

module {
  func.func @col_transform(%src: memref<64xf32>, %dst: memref<64xf32>) {
    %c1 = arith.constant 1 : index
    %c1_grid = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    gpu.launch blocks(%bx, %by, %bz) in (%gbx = %c1_grid, %gby = %c1, %gbz = %c1)
               threads(%tx, %ty, %tz) in (%gtx = %c64, %gty = %c1, %gtz = %c1) {
      %gid = arith.addi %bx, %tx : index

      %in_bounds = arith.cmpi ult, %gid, %c64 : index
      scf.if %in_bounds {
        %d0 = arith.constant 8 : index
        %d1 = arith.constant 8 : index
        %layout = lego.col [%d0, %d1] : !lego.layout
        %row = lego.row [%d0, %d1] : !lego.layout
        %order = lego.order_by(%row) : !lego.layout
        %identity = lego.group_by [%d0, %d1](%order) : !lego.layout

        %i0, %i1 = lego.apply_inverse %layout(%gid) : !lego.layout -> index, index
        %out_idx = lego.apply %identity(%i0, %i1) : !lego.layout

        %val = memref.load %src[%gid] : memref<64xf32>
        memref.store %val, %dst[%out_idx] : memref<64xf32>
      }
      gpu.terminator
    }
    return
  }
}

// CHECK: lego.spirv_binary
