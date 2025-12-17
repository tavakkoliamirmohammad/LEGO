cd mlir_generated/transpose

echo "Running Matrix Transpose"
bash run.sh > ../../result/mlir/transpose.txt

echo "Plotting Matrix Transpose"
cd ../../plots
python3 mlir_transpose.py
cd ..