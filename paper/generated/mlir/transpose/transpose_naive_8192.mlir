module {
  func.func @main() {
    %c8_i32 = arith.constant 8 : i32
    %c262144_i32 = arith.constant 262144 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c65536 = arith.constant 65536 : index
    affine.for %arg0 = 0 to 125 {
      %0 = gpu.wait async
      %memref, %asyncToken = gpu.alloc async [%0] () : memref<67108864xf32>
      %memref_0, %asyncToken_1 = gpu.alloc async [%asyncToken] () : memref<67108864xf32>
      %alloc = memref.alloc() : memref<67108864xf32>
      affine.for %arg1 = 0 to 67108864 {
        %4 = arith.index_cast %arg1 : index to i32
        %5 = arith.sitofp %4 : i32 to f32
        memref.store %5, %alloc[%arg1] : memref<67108864xf32>
      }
      %1 = gpu.memcpy async [%asyncToken_1] %memref_0, %alloc : memref<67108864xf32>, memref<67108864xf32>
      gpu.launch blocks(%arg1, %arg2, %arg3) in (%arg7 = %c65536, %arg8 = %c1, %arg9 = %c1) threads(%arg4, %arg5, %arg6) in (%arg10 = %c256, %arg11 = %c1, %arg12 = %c1) {
        memref.assume_alignment %memref, 128 : memref<67108864xf32>
        memref.assume_alignment %memref_0, 128 : memref<67108864xf32>
        %block_id_x = gpu.block_id  x
        %4 = arith.index_cast %block_id_x : index to i32
        %thread_id_x = gpu.thread_id  x
        %5 = arith.index_cast %thread_id_x : index to i32
        %6 = arith.remui %4, %c256_i32 : i32
        %7 = arith.muli %6, %c32_i32 : i32
        %8 = arith.divui %5, %c32_i32 : i32
        %9 = arith.muli %8, %c32_i32 : i32
        %10 = arith.remui %5, %c32_i32 : i32
        %11 = arith.addi %9, %10 : i32
        %12 = arith.divui %11, %c32_i32 : i32
        %13 = arith.muli %12, %c32_i32 : i32
        %14 = arith.addi %13, %10 : i32
        %15 = arith.divui %14, %c32_i32 : i32
        %16 = arith.muli %15, %c8192_i32 : i32
        %17 = arith.divui %4, %c256_i32 : i32
        %18 = arith.muli %17, %c256_i32 : i32
        %19 = arith.addi %18, %6 : i32
        %20 = arith.divui %19, %c256_i32 : i32
        %21 = arith.muli %20, %c256_i32 : i32
        %22 = arith.addi %21, %6 : i32
        %23 = arith.divui %22, %c256_i32 : i32
        %24 = arith.muli %23, %c262144_i32 : i32
        %25 = arith.addi %7, %16 : i32
        %26 = arith.muli %17, %c32_i32 : i32
        %27 = arith.muli %10, %c8192_i32 : i32
        %28 = arith.muli %6, %c262144_i32 : i32
        affine.for %arg13 = 0 to 4 {
          %29 = arith.index_cast %arg13 : index to i32
          %30 = arith.muli %29, %c65536_i32 : i32
          %31 = arith.addi %25, %30 : i32
          %32 = arith.addi %31, %24 : i32
          %33 = arith.addi %32, %10 : i32
          %34 = arith.index_cast %33 : i32 to index
          %35 = memref.load %memref_0[%34] : memref<67108864xf32>
          %36 = arith.muli %29, %c8_i32 : i32
          %37 = arith.addi %36, %26 : i32
          %38 = arith.addi %37, %27 : i32
          %39 = arith.addi %38, %28 : i32
          %40 = arith.addi %39, %8 : i32
          %41 = arith.index_cast %40 : i32 to index
          memref.store %35, %memref[%41] : memref<67108864xf32>
        }
        gpu.terminator
      }
      %2 = gpu.dealloc async [%1] %memref : memref<67108864xf32>
      %3 = gpu.dealloc async [%2] %memref_0 : memref<67108864xf32>
    }
    return
  }
}

