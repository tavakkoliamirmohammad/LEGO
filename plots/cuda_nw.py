#!/usr/bin/env python
import re
import numpy as np
import matplotlib.pyplot as plt

with open("../result/cuda/nw.txt", "r") as file:
    data = file.read()

pattern = r"Running tests with SIZE = (\d+).*?original.*?Execution time:\s*([\d.]+).*?antidiag.*?Execution time:\s*([\d.]+)"
matches = re.findall(pattern, data, re.DOTALL)

# Lists to hold parsed data.
matrix_sizes = []
original_times = []
antidiag_times = []

for size_str, orig_time_str, antidiag_time_str in matches:
    matrix_sizes.append(int(size_str))
    original_times.append(float(orig_time_str))
    antidiag_times.append(float(antidiag_time_str))

# Convert lists to NumPy arrays.
matrix_sizes = np.array(matrix_sizes)[::-1]
original_times = np.array(original_times)[::-1]
antidiag_times = np.array(antidiag_times)[::-1]

# Compute speedup (Original / Antidiag)
speedup = original_times / antidiag_times

# Use indices for equally spaced groups, with tick labels set to matrix_sizes.
indices = np.arange(len(matrix_sizes))
bar_width = 0.35  # adjust bar width if needed

plt.figure(figsize=(5, 3))
# plt.bar(indices, speedup, bar_width, label='Speedup')

# Connect the speedup points with a line.
plt.bar(indices, speedup, label='Antidiagonal')

# Create a dashed horizontal line at y = 1.
plt.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='Original')

plt.xlabel('Matrix Size')
plt.ylabel('Speedup')
# plt.title('Speedup by Matrix Size')
plt.xticks(indices, matrix_sizes)
plt.grid(True, axis='y', linestyle='--', linewidth=0.3)
plt.ylim(bottom=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('cuda_nw.pdf')
# plt.show()

print("Matrix Sizes:      ", matrix_sizes)
print("Original Times:    ", original_times)
print("Antidiag Times:    ", antidiag_times)
print("Speedup (Orig/Antidiag):", speedup)
