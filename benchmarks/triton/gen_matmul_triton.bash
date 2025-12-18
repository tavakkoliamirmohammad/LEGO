export PYTHONPATH=../..:$PYTHONPATH
cd "$(dirname "$0")"
mkdir -p ../../triton_generated/matmul

python3 matmul_sympy.py --NTA --NTB > ./../../triton_generated/matmul/matmul_NTA_NTB.py
python3 matmul_sympy.py --NTA --TB > ./../../triton_generated/matmul/matmul_NTA_TB.py
python3 matmul_sympy.py --TA --NTB > ./../../triton_generated/matmul/matmul_TA_NTB.py
python3 matmul_sympy.py --TA --TB > ./../../triton_generated/matmul/matmul_TA_TB.py