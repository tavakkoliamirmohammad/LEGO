export PYTHONPATH=../..:$PYTHONPATH
cd "$(dirname "$0")"
cd ../../triton_generated

echo "Running Group GEMM"
cd grouped_gemm
bash run.sh  > ../../result/triton/group_gemm.txt

echo "Running GEMM"
cd ../matmul
bash run.sh  > ../../result/triton/matmul.txt

echo "Running Softmax"
cd ../softmax
bash run.sh  > ../../result/triton/softmax.txt

echo "Running Layernorm"
cd ../layernorm
bash run.sh  > ../../result/triton/layernorm.txt


cd ../../plots
echo "Plotting Triton Benchmarks"
python3 new_triton.py
cd ../../
