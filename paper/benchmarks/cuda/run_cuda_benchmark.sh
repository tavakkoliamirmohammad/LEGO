export PYTHONPATH=../../../python:$PYTHONPATH
cd "$(dirname "$0")"
cd nw

echo "Running NW"
bash run.sh > ../../../results/cuda/nw.txt

cd ..
echo "Running LUD"
bash run_lud_cuda.sh  > ../../results/cuda/lud.txt

echo "Running Bricks"
bash gen_bricks_f3d.bash  > ../../results/cuda/bricks-f3d.txt
bash gen_bricks_laplace.bash  > ../../results/cuda/bricks-laplace.txt


cd ../../plots
echo "Plotting NW"
python3 cuda_nw.py
echo "Plotting LUD"
python3 lud.py
echo "Plotting Bricks"
python3 bricks.py

cd ../../../