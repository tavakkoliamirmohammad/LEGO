SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../../.."
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT/build/python_packages/lego:$REPO_ROOT/build/python:$PYTHONPATH"
cd "$SCRIPT_DIR"

cd lud
echo "----"
make ncu-export-roofline
cd ..

cd bricks
echo "----"
make ncu-export-roofline
cd ..

cd bricks-laplace
echo "----"
make ncu-export-roofline
cd ..

echo "Generating roofline data for LUD..."
cd lud
bash run_ncu.sh
cd ..

echo "Generating roofline data for Stencil..."
cd bricks_reports
bash gen_bricks_roofline.bash
cd ../..