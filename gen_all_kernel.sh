echo "Generating all kernel configurations..."
echo "Generating all Triton kernels..."
bash gen_all_triton.sh
echo "Generating all CUDA kernels..."
bash gen_all_cuda.sh
echo "Generating all MLIR kernels..."
bash gen_transpose_mlir.bash
echo "All kernel configurations generated."