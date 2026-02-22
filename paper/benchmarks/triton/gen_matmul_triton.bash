export PYTHONPATH=../../../python:$PYTHONPATH
cd "$(dirname "$0")"
mkdir -p ../../generated/triton/matmul

python3 matmul_sympy.py --NTA --NTB > ./../../generated/triton/matmul/matmul_NTA_NTB.py
python3 matmul_sympy.py --NTA --TB > ./../../generated/triton/matmul/matmul_NTA_TB.py
python3 matmul_sympy.py --TA --NTB > ./../../generated/triton/matmul/matmul_TA_NTB.py
python3 matmul_sympy.py --TA --TB > ./../../generated/triton/matmul/matmul_TA_TB.py