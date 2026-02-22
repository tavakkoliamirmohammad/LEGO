export PYTHONPATH=../../../python:$PYTHONPATH
cd "$(dirname "$0")"
mkdir -p ../../generated/triton/grouped_gemm

python3 grouped_gemm_sympy.py > ./../../generated/triton/grouped_gemm/grouped_gemm.py
