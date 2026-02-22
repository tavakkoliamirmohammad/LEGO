export PYTHONPATH=../../../python:$PYTHONPATH
cd "$(dirname "$0")"
mkdir -p ../../generated/triton/softmax

python3 softmax_sympy.py > ./../../generated/triton/softmax/softmax.py
