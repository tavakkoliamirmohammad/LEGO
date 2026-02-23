#!/bin/bash
export PYTHONPATH=../../..:$PYTHONPATH
echo "----"
make run-bricks-r1
echo "----"
make run-bricks-r2
echo "----"
make run-bricks-r3
echo "----"
make run-bricks-r4
