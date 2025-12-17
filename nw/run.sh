#!/bin/bash
# Define an array of SIZE values.
sizes=(32768 16384 8192 4096 2048 1024)

# Loop over each SIZE value.
for SIZE in "${sizes[@]}"; do
  echo "====================================="
  echo "Running tests with SIZE = $SIZE"
  echo "====================================="

  echo "original"
  # Pass the SIZE to make, e.g., by defining a make variable SIZE.
  make original SIZE="$SIZE"
  make run SIZE="$SIZE"

  echo "antidiag"
  make antidiag SIZE="$SIZE"
  make run SIZE="$SIZE"
  
  echo ""
done
