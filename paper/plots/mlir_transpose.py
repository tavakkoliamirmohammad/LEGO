#!/usr/bin/env python3
import re
from collections import defaultdict

with open("../result/mlir/transpose.txt", "r", encoding="utf-8") as file:
    data = file.read()

pattern = r'(.*?)\s*:\s*(\d+):\s*\nAverage:\s*([\d.]+)\s*ms'
matches = re.findall(pattern, data)

methods = [
    "Transpose Naive",
    "CUDA Transpose Naive",
    "Transpose Shared",
    "CUDA Coalesced Naive",
]

# Collect averages by method and size (ms)
records = defaultdict(dict)
all_sizes = set()
for method, size, avg in matches:
    method = method.strip()
    size = int(size)
    avg = float(avg)   # ms
    if method in methods:
        records[method][size] = avg
        all_sizes.add(size)
    else:
        print("Warning: unrecognized method:", method)

sizes_sorted = sorted(all_sizes)


def throughput_ops_per_s(ms: float, M: int) -> float:
    """Throughput in GB/s, given ms in milliseconds and M as matrix size."""
    bytes_moved = 2.0 * M * M * 4        # 2 matrices * M*M elements * 4 bytes each
    gb_moved = bytes_moved / 1e9         # convert bytes → GB
    seconds = ms / 1e3                   # convert ms → s
    return gb_moved / seconds


# ---------- Print throughput table to file ----------
outfile = "mlir_transpose.txt"

with open(outfile, "w") as f:
    f.write("\nThroughput (GB/s)\n\n")

    hdr = ["Size"] + methods
    colw = [6, 20, 24, 20, 26]

    header_line = "  ".join(f"{h:>{w}}" for h, w in zip(hdr, colw))
    f.write(header_line + "\n")
    f.write("-" * sum(colw) + "-" * (len(colw) - 1) + "\n")

    for M in sizes_sorted:
        row = [f"{M:>{colw[0]}}"]
        for mi, method in enumerate(methods, start=1):
            ms = records[method].get(M)
            if ms is None:
                cell = "N/A"
            else:
                thr = throughput_ops_per_s(ms, M)
                cell = f"{thr:.4}"
            row.append(f"{cell:>{colw[mi]}}")
        f.write("  ".join(row) + "\n")

print(f"Throughput table written to {outfile}")
