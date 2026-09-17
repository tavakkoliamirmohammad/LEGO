// RUN: lego-opt %s --lego-to-nvvm='chip=sm_86 features=+ptx78 format=assembly' | FileCheck %s
// Vertical slice for tensor-core kernels: cp.async, ldmatrix and mma.sync must
// survive the LEGO NVVM pipeline and appear in PTX.
// CHECK: cp.async
// CHECK: ldmatrix
// CHECK: mma.sync
module {
  func.func @tc(%a: memref<64x64xf16>, %out: memref<64x8xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gbx = %c1, %gby = %c1, %gbz = %c1)
               threads(%tx, %ty, %tz) in (%gtx = %c32, %gty = %c1, %gtz = %c1) {
      %smem = memref.alloca() : memref<64x64xf16, 3>
      %tok = nvgpu.device_async_copy %a[%tx, %c0], %smem[%tx, %c0], 8 : memref<64x64xf16> to memref<64x64xf16, 3>
      %grp = nvgpu.device_async_create_group %tok
      nvgpu.device_async_wait %grp
      gpu.barrier
      %fa = nvgpu.ldmatrix %smem[%c0, %c0] {transpose = false, numTiles = 4 : i32} : memref<64x64xf16, 3> -> vector<4x2xf16>
      %fb = nvgpu.ldmatrix %smem[%c0, %c0] {transpose = true, numTiles = 2 : i32} : memref<64x64xf16, 3> -> vector<2x2xf16>
      %cst = arith.constant dense<0.0> : vector<2x2xf32>
      %d = nvgpu.mma.sync (%fa, %fb, %cst) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
      %v = vector.extract %d[0, 0] : f32 from vector<2x2xf32>
      memref.store %v, %out[%tx, %c0] : memref<64x8xf32>
      gpu.terminator
    }
    return
  }
}
