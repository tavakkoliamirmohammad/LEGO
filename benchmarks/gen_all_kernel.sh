export PYTHONPATH=..:$PYTHONPATH
cd "$(dirname "$0")"
echo "Generating all kernel configurations..."
echo "Generating all Triton kernels..."
bash triton/gen_all_triton.sh
echo "Generating all CUDA kernels..."
bash cuda/gen_all_cuda.sh
echo "Generating all MLIR kernels..."
bash mlir/gen_transpose_mlir.bash
echo "All kernel configurations generated."