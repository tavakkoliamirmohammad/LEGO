// RUN: lego-opt %s --lego-to-nvvm | mlir-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime,%mlir_c_runner_utils,%mlir_runner_utils \
// RUN:   --entry-point-result=void | FileCheck %s

// End-to-end GPU transpose test with host-side verification.
// Transposes a 4x4 matrix on the GPU using LEGO layout ops,
// then checks the result on the host.
//
// Input A:                Expected B = A^T:
//   0  1  2  3             0  4  8 12
//   4  5  6  7             1  5  9 13
//   8  9 10 11             2  6 10 14
//  12 13 14 15             3  7 11 15

module attributes {gpu.container_module} {

  gpu.module @transpose_module {
    // Simple transpose kernel: B[j*4+i] = A[i*4+j]
    // Uses LEGO layout ops to compute indices.
    gpu.func @transpose_kernel(%A: memref<16xf32>, %B: memref<16xf32>)
        kernel attributes {known_block_size = array<i32: 16, 1, 1>} {
      %tid = gpu.thread_id x

      // Decompose flat tid into (row, col) via Row(4,4) layout
      %c4_0 = arith.constant 4 : index
      %c4_1 = arith.constant 4 : index
      %row_layout = lego.row [%c4_0, %c4_1] : !lego.layout
      %row, %col = lego.apply_inverse %row_layout(%tid) : !lego.layout -> index, index

      // Compute transposed index: col*4 + row
      %col_layout = lego.col [%c4_0, %c4_1] : !lego.layout
      %t_idx = lego.apply %col_layout(%row, %col) : !lego.layout

      // B[t_idx] = A[tid]
      %val = memref.load %A[%tid] : memref<16xf32>
      memref.store %val, %B[%t_idx] : memref<16xf32>
      gpu.return
    }
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index

    // Allocate host matrices
    %A = memref.alloc() : memref<16xf32>
    %B = memref.alloc() : memref<16xf32>

    // Fill A = [0, 1, 2, ..., 15]
    scf.for %i = %c0 to %c16 step %c1 {
      %idx_i32 = arith.index_cast %i : index to i32
      %val = arith.sitofp %idx_i32 : i32 to f32
      memref.store %val, %A[%i] : memref<16xf32>
    }

    // Zero B
    %zero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c16 step %c1 {
      memref.store %zero, %B[%i] : memref<16xf32>
    }

    // Register for GPU access
    %Au = memref.cast %A : memref<16xf32> to memref<*xf32>
    %Bu = memref.cast %B : memref<16xf32> to memref<*xf32>
    gpu.host_register %Au : memref<*xf32>
    gpu.host_register %Bu : memref<*xf32>

    // Launch transpose kernel: 1 block of 16 threads
    gpu.launch_func @transpose_module::@transpose_kernel
        blocks in (%c1, %c1, %c1)
        threads in (%c16, %c1, %c1)
        args(%A : memref<16xf32>, %B : memref<16xf32>)

    // Print result (B should be the transpose of A)
    // A as 4x4 row-major: [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
    // B = A^T:             [[0,4,8,12],[1,5,9,13],[2,6,10,14],[3,7,11,15]]
    // Flat B:              [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    func.call @printMemrefF32(%Bu) : (memref<*xf32>) -> ()

    memref.dealloc %A : memref<16xf32>
    memref.dealloc %B : memref<16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>)
}

// CHECK: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
