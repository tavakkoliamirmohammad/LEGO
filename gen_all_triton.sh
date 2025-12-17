echo "Generating all Triton kernels..."

echo "Matmul..."
bash gen_matmul_triton.bash
echo "Grouped Gemm..."
bash gen_grouped_gemm_triton.bash
echo "Softmax..."
bash gen_softmax_triton.bash
echo "Layernorm..."
bash gen_layernorm_triton.bash