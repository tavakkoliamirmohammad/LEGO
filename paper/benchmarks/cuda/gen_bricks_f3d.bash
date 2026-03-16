SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../../.."
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT/build/python_packages/lego:$REPO_ROOT/build/python:$PYTHONPATH"
cd "$SCRIPT_DIR"
cd bricks
echo "----"
make run-bricks-r1
echo "----"
make run-bricks-r2
cd ..