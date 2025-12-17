mkdir -p triton_generated/layernorm

python3 layernorm_triton.py > ./triton_generated/layernorm/layernorm.py
