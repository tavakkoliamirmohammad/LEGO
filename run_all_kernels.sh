echo "Run MLIR Benchmarks"
bash run_mlir_benchmark.sh
echo "Run Triton Benchmarks"
bash run_triton_benchmark.sh
echo "Run CUDA Benchmarks"
bash run_cuda_benchmark.sh
echo "All benchmarks completed."

echo "Generating Roofline Data for CUDA Kernels"
bash gen_roofline_cuda.sh
echo "Roofline data generation completed."

echo "Extracting Figures"
bash extract_figure.sh
echo "Figures extracted."