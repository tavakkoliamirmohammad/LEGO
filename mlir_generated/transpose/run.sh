#!/bin/bash

# Enable strict error handling
set -euo pipefail
IFS=$'\n\t'
# Enable alias expansion
shopt -s expand_aliases

# Function to process a single MLIR file
process_file() {
    local N="$1"

    echo "Transpose Naive: $N: "

   $MLIR_BUILD_FOLDER/bin/mlir-opt --canonicalize --cse --arith-expand --loop-invariant-code-motion \
        --lower-affine \
        -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 opt-level=3" \
        "transpose_naive_${N}.mlir" | \
    ncu --metrics gpu__time_duration \
       $MLIR_BUILD_FOLDER/bin/mlir-cpu-runner \
        --shared-libs=$MLIR_BUILD_FOLDER/lib/libmlir_cuda_runtime.so \
        --shared-libs=$MLIR_BUILD_FOLDER/lib/libmlir_c_runner_utils.so \
        --entry-point-result=void \
        -O3 | \
    grep "gpu__time_duration.avg" | \
    awk '{
        # Skip the first 25 warmup runs
        if (NR <= 25) next;
        # Determine conversion factor to milliseconds
        if ($2=="ms")      conv = 1;
        else if ($2=="us") conv = 0.001;
        else if ($2=="ns") conv = 0.000001;
        else               conv = 1;
        # Accumulate sum (in ms) and count
        sum += $3 * conv; count++
    }
    END {
        if (count > 0) {
            printf("Average: %.6f ms\n", sum / count);
        }
    }'

    echo "CUDA Transpose Naive: $N: "
    nvcc ./transpose.cu -o transpose.o && \
    ncu -k transposeNaive --metrics gpu__time_duration transpose.o $N | \
    grep "gpu__time_duration.avg" | \
    awk '{
        if (NR <= 25) next;
        if ($2=="ms")      conv = 1;
        else if ($2=="us") conv = 0.001;
        else if ($2=="ns") conv = 0.000001;
        else               conv = 1;
        sum += $3 * conv; count++
    }
    END {
        if (count > 0) {
            printf("Average: %.6f ms\n", sum / count);
        }
    }'

    echo "Transpose Shared: $N: "
   $MLIR_BUILD_FOLDER/bin/mlir-opt --canonicalize --cse --arith-expand --loop-invariant-code-motion \
        --lower-affine \
        -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 opt-level=3" \
        "transpose_smem_${N}.mlir" | \
    ncu --metrics gpu__time_duration \
       $MLIR_BUILD_FOLDER/bin/mlir-cpu-runner \
        --shared-libs=$MLIR_BUILD_FOLDER/lib/libmlir_cuda_runtime.so \
        --shared-libs=$MLIR_BUILD_FOLDER/lib/libmlir_c_runner_utils.so \
        --entry-point-result=void \
        -O3 | \
    grep "gpu__time_duration.avg" | \
    awk '{
        if (NR <= 25) next;
        if ($2=="ms")      conv = 1;
        else if ($2=="us") conv = 0.001;
        else if ($2=="ns") conv = 0.000001;
        else               conv = 1;
        sum += $3 * conv; count++
    }
    END {
        if (count > 0) {
            printf("Average: %.6f ms\n", sum / count);
        }
    }'

    echo "CUDA Coalesced Naive: $N: "
    nvcc ./transpose.cu -o transpose.o && \
    ncu -k transposeCoalesced --metrics gpu__time_duration transpose.o $N | \
    grep "gpu__time_duration.avg" | \
    awk '{
        if (NR <= 25) next;
        if ($2=="ms")      conv = 1;
        else if ($2=="us") conv = 0.001;
        else if ($2=="ns") conv = 0.000001;
        else               conv = 1;
        sum += $3 * conv; count++
    }
    END {
        if (count > 0) {
            printf("Average: %.6f ms\n", sum / count);
        }
    }'

    echo "------------------------------------------"
}

# Loop through powers of 2 from 256 to 8192
for (( size=2048; size<=8192; size*=2 )); do
    process_file "$size"
done
