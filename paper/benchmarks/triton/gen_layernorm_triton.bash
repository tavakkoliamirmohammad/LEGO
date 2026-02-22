export PYTHONPATH=../../../python:$PYTHONPATH
cd "$(dirname "$0")"
mkdir -p ../../generated/triton/layernorm

python3 layernorm_triton.py > ./../../generated/triton/layernorm/layernorm.py
