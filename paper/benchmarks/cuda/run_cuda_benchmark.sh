SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$SCRIPT_DIR/../.."

echo "Running NW"
cd $BASE_DIR/generated/cuda/nw
bash run.sh > $BASE_DIR/results/cuda/nw.txt
cd -

echo "Running LUD"
cd $BASE_DIR/generated/cuda/lud
bash run.sh > $BASE_DIR/results/cuda/lud.txt
cd -

echo "Running Bricks f3d"
cd $BASE_DIR/generated/cuda/bricks
bash run.sh > $BASE_DIR/results/cuda/bricks-f3d.txt
cd -

echo "Running Bricks laplace"
cd $BASE_DIR/generated/cuda/bricks-laplace
bash run.sh > $BASE_DIR/results/cuda/bricks-laplace.txt
cd -


cd $SCRIPT_DIR/../plots
echo "Plotting NW"
python3 cuda_nw.py
echo "Plotting LUD"
python3 lud.py
echo "Plotting Bricks"
python3 bricks.py

cd $SCRIPT_DIR