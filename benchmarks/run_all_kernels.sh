export PYTHONPATH=..:$PYTHONPATH
cd "$(dirname "$0")"
echo "Run MLIR Benchmarks"
bash mlir/run_mlir_benchmark.sh
echo "Run Triton Benchmarks"
bash triton/run_triton_benchmark.sh
echo "Run CUDA Benchmarks"
bash cuda/run_cuda_benchmark.sh
echo "All benchmarks completed."

echo "Generating Roofline Data for CUDA Kernels"
bash cuda/gen_roofline_cuda.sh
echo "Roofline data generation completed."

echo "Extracting Figures"
bash ../extract_figure.sh
echo "Figures extracted."