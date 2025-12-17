export PYTHONPATH=../..:$PYTHONPATH
cd "$(dirname "$0")"
echo "Generating all CUDA kernels..."
echo "NW..."
python3 nw_sympy.py
echo "LUD..."
python3 lud.py
echo "Stencil Star..."
python3 bricks_laplasian.py
echo "Stencil Cube..."
python3 bricks_f3d.py