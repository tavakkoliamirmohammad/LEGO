export PYTHONPATH=../../../python:$PYTHONPATH
cd "$(dirname "$0")"
mkdir -p ../../generated/mlir/transpose

MLIR_OPT_PATH=$MLIR_BUILD_FOLDER/bin/mlir-opt
# Loop through powers of 2 from 32 to 8192
for (( size=256; size<=8192; size*=2 )); do
    # Execute the first command with N replaced by size
    PYTHONPATH=../../../python:$MLIR_BUILD_FOLDER/tools/mlir/python_packages/mlir_core python3 transpose_naive.py "$size" "$size" | $MLIR_OPT_PATH --canonicalize --cse -loop-invariant-code-motion --cse > "../../generated/mlir/transpose/transpose_naive_${size}.mlir"
    
    # Execute the second command with N replaced by size
    PYTHONPATH=../../../python:$MLIR_BUILD_FOLDER/tools/mlir/python_packages/mlir_core python3 transpose_smem.py "$size" "$size" | $MLIR_OPT_PATH  --canonicalize --cse -loop-invariant-code-motion > "../../generated/mlir/transpose/transpose_smem_${size}.mlir"
done

