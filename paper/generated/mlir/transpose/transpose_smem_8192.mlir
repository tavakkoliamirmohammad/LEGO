module {
  func.func @main() {
    %c8192 = arith.constant 8192 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %c65536 = arith.constant 65536 : index
    %c67108864 = arith.constant 67108864 : index
    %c0 = arith.constant 0 : index
    %c125 = arith.constant 125 : index
    %c1 = arith.constant 1 : index
    scf.for %arg0 = %c0 to %c125 step %c1 {
      %0 = gpu.wait async
      %memref, %asyncToken = gpu.alloc async [%0] () : memref<67108864xf32>
      %alloc = memref.alloc() : memref<67108864xf32>
      %memref_0, %asyncToken_1 = gpu.alloc async [%asyncToken] () : memref<67108864xf32>
      scf.for %arg1 = %c0 to %c67108864 step %c1 {
        %4 = arith.index_cast %arg1 : index to i32
        %5 = arith.sitofp %4 : i32 to f32
        memref.store %5, %alloc[%arg1] : memref<67108864xf32>
      }
      %1 = gpu.memcpy async [%asyncToken_1] %memref, %alloc : memref<67108864xf32>, memref<67108864xf32>
      gpu.launch blocks(%arg1, %arg2, %arg3) in (%arg7 = %c65536, %arg8 = %c1, %arg9 = %c1) threads(%arg4, %arg5, %arg6) in (%arg10 = %c256, %arg11 = %c1, %arg12 = %c1) workgroup(%arg13 : memref<1024xf32, 3>) {
        %block_id_x = gpu.block_id  x
        %thread_id_x = gpu.thread_id  x
        %4 = arith.divui %block_id_x, %c256 : index
        %5 = arith.remui %block_id_x, %c256 : index
        %6 = arith.divui %thread_id_x, %c32 : index
        %7 = arith.remui %thread_id_x, %c32 : index
        scf.for %arg14 = %c0 to %c4 step %c1 {
          %8 = arith.muli %arg14, %c8 : index
          %9 = arith.addi %6, %8 : index
          %10 = arith.muli %4, %c32 : index
          %11 = arith.addi %9, %10 : index
          %12 = arith.muli %5, %c32 : index
          %13 = arith.addi %7, %12 : index
          %14 = arith.muli %11, %c8192 : index
          %15 = arith.addi %13, %14 : index
          %16 = memref.load %memref[%15] : memref<67108864xf32>
          %17 = arith.muli %9, %c32 : index
          %18 = arith.addi %7, %17 : index
          memref.store %16, %arg13[%18] : memref<1024xf32, 3>
        }
        gpu.barrier
        scf.for %arg14 = %c0 to %c4 step %c1 {
          %8 = arith.muli %arg14, %c8 : index
          %9 = arith.addi %6, %8 : index
          %10 = arith.muli %7, %c32 : index
          %11 = arith.addi %9, %10 : index
          %12 = memref.load %arg13[%11] : memref<1024xf32, 3>
          %13 = arith.muli %5, %c32 : index
          %14 = arith.addi %9, %13 : index
          %15 = arith.muli %4, %c32 : index
          %16 = arith.addi %7, %15 : index
          %17 = arith.muli %14, %c8192 : index
          %18 = arith.addi %16, %17 : index
          memref.store %12, %memref_0[%18] : memref<67108864xf32>
        }
        gpu.terminator
      }
      %2 = gpu.dealloc async [%1] %memref : memref<67108864xf32>
      %3 = gpu.dealloc async [%2] %memref_0 : memref<67108864xf32>
    }
    return
  }
}

