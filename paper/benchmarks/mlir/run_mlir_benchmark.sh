export PYTHONPATH=../../../python:$PYTHONPATH
cd "$(dirname "$0")"
cd ../../generated/mlir/transpose

echo "Running Matrix Transpose"
bash run.sh > ../../../results/mlir/transpose.txt

echo "Plotting Matrix Transpose"
cd ../../plots
python3 mlir_transpose.py
cd ../../../