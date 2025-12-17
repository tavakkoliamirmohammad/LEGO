export PYTHONPATH=../..:$PYTHONPATH
cd "$(dirname "$0")"
cd bricks
echo "----"
make run-bricks-r1
echo "----"
make run-bricks-r2
cd ..