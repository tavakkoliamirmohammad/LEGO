#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../../../.."
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT/build/python_packages/lego:$REPO_ROOT/build/python:$PYTHONPATH"
echo "----"
make run-bricks-r1
echo "----"
make run-bricks-r2
echo "----"
make run-bricks-r3
echo "----"
make run-bricks-r4
