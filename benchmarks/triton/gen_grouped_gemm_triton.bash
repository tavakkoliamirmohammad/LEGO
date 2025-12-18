export PYTHONPATH=../..:$PYTHONPATH
cd "$(dirname "$0")"
mkdir -p ../../triton_generated/grouped_gemm

python3 grouped_gemm_sympy.py > ./../../triton_generated/grouped_gemm/grouped_gemm.py
