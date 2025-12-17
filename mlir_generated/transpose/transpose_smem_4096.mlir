module {
  func.func @main() {
    %c8_i32 = arith.constant 8 : i32
    %c256_i32 = arith.constant 256 : i32
    %c131072_i32 = arith.constant 131072 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c16384 = arith.constant 16384 : index
    affine.for %arg0 = 0 to 125 {
      %0 = gpu.wait async
      %memref, %asyncToken = gpu.alloc async [%0] () : memref<16777216xf32>
      %alloc = memref.alloc() : memref<16777216xf32>
      %memref_0, %asyncToken_1 = gpu.alloc async [%asyncToken] () : memref<16777216xf32>
      affine.for %arg1 = 0 to 16777216 {
        %4 = arith.index_cast %arg1 : index to i32
        %5 = arith.sitofp %4 : i32 to f32
        memref.store %5, %alloc[%arg1] : memref<16777216xf32>
      }
      %1 = gpu.memcpy async [%asyncToken_1] %memref, %alloc : memref<16777216xf32>, memref<16777216xf32>
      gpu.launch blocks(%arg1, %arg2, %arg3) in (%arg7 = %c16384, %arg8 = %c1, %arg9 = %c1) threads(%arg4, %arg5, %arg6) in (%arg10 = %c256, %arg11 = %c1, %arg12 = %c1) workgroup(%arg13 : memref<1024xf32, 3>) {
        memref.assume_alignment %memref, 128 : memref<16777216xf32>
        memref.assume_alignment %memref_0, 128 : memref<16777216xf32>
        %block_id_x = gpu.block_id  x
        %4 = arith.index_cast %block_id_x : index to i32
        %thread_id_x = gpu.thread_id  x
        %5 = arith.index_cast %thread_id_x : index to i32
        %6 = arith.remui %4, %c128_i32 : i32
        %7 = arith.muli %6, %c32_i32 : i32
        %8 = arith.divui %5, %c32_i32 : i32
        %9 = arith.muli %8, %c32_i32 : i32
        %10 = arith.remui %5, %c32_i32 : i32
        %11 = arith.addi %9, %10 : i32
        %12 = arith.divui %11, %c32_i32 : i32
        %13 = arith.muli %12, %c32_i32 : i32
        %14 = arith.addi %13, %10 : i32
        %15 = arith.divui %14, %c32_i32 : i32
        %16 = arith.muli %15, %c4096_i32 : i32
        %17 = arith.divui %4, %c128_i32 : i32
        %18 = arith.muli %17, %c128_i32 : i32
        %19 = arith.addi %18, %6 : i32
        %20 = arith.divui %19, %c128_i32 : i32
        %21 = arith.muli %20, %c128_i32 : i32
        %22 = arith.addi %21, %6 : i32
        %23 = arith.divui %22, %c128_i32 : i32
        %24 = arith.muli %23, %c131072_i32 : i32
        %25 = arith.addi %7, %16 : i32
        %26 = arith.muli %15, %c32_i32 : i32
        affine.for %arg14 = 0 to 4 {
          %42 = arith.index_cast %arg14 : index to i32
          %43 = arith.muli %42, %c32768_i32 : i32
          %44 = arith.addi %25, %43 : i32
          %45 = arith.addi %44, %24 : i32
          %46 = arith.addi %45, %10 : i32
          %47 = arith.index_cast %46 : i32 to index
          %48 = memref.load %memref[%47] : memref<16777216xf32>
          %49 = arith.muli %42, %c256_i32 : i32
          %50 = arith.addi %26, %49 : i32
          %51 = arith.addi %50, %10 : i32
          %52 = arith.index_cast %51 : i32 to index
          memref.store %48, %arg13[%52] : memref<1024xf32, 3>
        }
        gpu.barrier
        %27 = arith.remui %5, %c32_i32 : i32
        %28 = arith.muli %27, %c32_i32 : i32
        %29 = arith.divui %5, %c32_i32 : i32
        %30 = arith.divui %4, %c128_i32 : i32
        %31 = arith.muli %30, %c32_i32 : i32
        %32 = arith.muli %29, %c32_i32 : i32
        %33 = arith.addi %32, %27 : i32
        %34 = arith.divui %33, %c32_i32 : i32
        %35 = arith.muli %34, %c32_i32 : i32
        %36 = arith.addi %35, %27 : i32
        %37 = arith.divui %36, %c32_i32 : i32
        %38 = arith.muli %37, %c4096_i32 : i32
        %39 = arith.remui %4, %c128_i32 : i32
        %40 = arith.muli %39, %c131072_i32 : i32
        %41 = arith.addi %31, %38 : i32
        affine.for %arg14 = 0 to 4 {
          %42 = arith.index_cast %arg14 : index to i32
          %43 = arith.muli %42, %c8_i32 : i32
          %44 = arith.addi %43, %28 : i32
          %45 = arith.addi %44, %29 : i32
          %46 = arith.index_cast %45 : i32 to index
          %47 = memref.load %arg13[%46] : memref<1024xf32, 3>
          %48 = arith.muli %42, %c32768_i32 : i32
          %49 = arith.addi %41, %48 : i32
          %50 = arith.addi %49, %40 : i32
          %51 = arith.addi %50, %27 : i32
          %52 = arith.index_cast %51 : i32 to index
          memref.store %47, %memref_0[%52] : memref<16777216xf32>
        }
        gpu.terminator
      }
      %2 = gpu.dealloc async [%1] %memref : memref<16777216xf32>
      %3 = gpu.dealloc async [%2] %memref_0 : memref<16777216xf32>
    }
    return
  }
}

