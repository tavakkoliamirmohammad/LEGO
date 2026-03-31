// REQUIRES: amdgpu
// RUN: lego-opt %s --lego-to-rocdl='format=assembly' | FileCheck %s

// Test: LEGO layout ops inside gpu.launch lower through ROCDL pipeline.
// No AMD GPU hardware required — this is a compiler-only test.

module {
  func.func @transform(%src: memref<256xf32>, %dst: memref<256xf32>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index

    gpu.launch blocks(%bx, %by, %bz) in (%gbx = %c4, %gby = %c1, %gbz = %c1)
               threads(%tx, %ty, %tz) in (%gtx = %c64, %gty = %c1, %gtz = %c1) {
      %gid_base = arith.muli %bx, %gtx : index
      %gid = arith.addi %gid_base, %tx : index

      %in_bounds = arith.cmpi ult, %gid, %c256 : index
      scf.if %in_bounds {
        // LEGO layout ops
        %d0 = arith.constant 16 : index
        %d1 = arith.constant 16 : index
        %layout = lego.row [%d0, %d1] : !lego.layout
        %row2 = lego.row [%d0, %d1] : !lego.layout
        %order = lego.order_by(%row2) : !lego.layout
        %identity = lego.group_by [%d0, %d1](%order) : !lego.layout

        %i0, %i1 = lego.apply_inverse %layout(%gid) : !lego.layout -> index, index
        %out_idx = lego.apply %identity(%i0, %i1) : !lego.layout

        %val = memref.load %src[%gid] : memref<256xf32>
        memref.store %val, %dst[%out_idx] : memref<256xf32>
      }
      gpu.terminator
    }
    return
  }
}

// Verify the module lowered to ROCDL — the gpu.binary op should contain
// the compiled kernel object.
// CHECK: gpu.binary
