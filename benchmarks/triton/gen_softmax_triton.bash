export PYTHONPATH=../..:$PYTHONPATH
cd "$(dirname "$0")"
mkdir -p ../../triton_generated/softmax

python3 softmax_sympy.py > ./../../triton_generated/softmax/softmax.py
