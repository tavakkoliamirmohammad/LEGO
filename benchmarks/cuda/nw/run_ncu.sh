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
  make run-ncu-original SIZE="$SIZE"

  echo "antidiag"
  make run-ncu-antidiag SIZE="$SIZE"
  
  echo ""
done

echo "Exporting NCU data..."
make ncu-export-roofline
echo "Generating roofline plot..."
python3 plot_roofline.py
