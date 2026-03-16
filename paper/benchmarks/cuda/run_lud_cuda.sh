SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../../.."
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT/build/python_packages/lego:$REPO_ROOT/build/python:$PYTHONPATH"
cd "$SCRIPT_DIR"

cd lud
echo "----"
make run-lego-16
echo "----"
make run-lego-8
echo "----"
make run-lego-4
echo "----"
make run-lego-2
echo "----"
make run-orig-32
echo "----"
make run-orig-16
cd ..