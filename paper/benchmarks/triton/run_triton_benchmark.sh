export PYTHONPATH=../../../python:$PYTHONPATH
cd "$(dirname "$0")"
cd ../../generated/triton

echo "Running Group GEMM"
cd grouped_gemm
bash run.sh  > ../../../results/triton/group_gemm.txt

echo "Running GEMM"
cd ../matmul
bash run.sh  > ../../../results/triton/matmul.txt

echo "Running Softmax"
cd ../softmax
bash run.sh  > ../../../results/triton/softmax.txt

echo "Running Layernorm"
cd ../layernorm
bash run.sh  > ../../../results/triton/layernorm.txt


cd ../../plots
echo "Plotting Triton Benchmarks"
python3 new_triton.py
cd ../../
