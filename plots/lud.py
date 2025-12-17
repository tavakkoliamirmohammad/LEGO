import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# --- Load log ---
log_path = Path("../result/cuda/lud.txt")
with log_path.open("r") as f:
    raw_log = f.read()

# --- Helpers ---
def detect_variant(nvcc_line: str) -> str:
    """
    Map an 'nvcc ...' line to a friendly variant label.
    Supports:
      -DRD_WG_SIZE_0=16  -> ORIG-16 (WG=16)
      -DRD_WG_SIZE_0=32  -> ORIG-32 (WG=32)
      -DLEGO_R<k>=1      -> LEGO_R<k> (WG=<k>)
    """
    if "-DRD_WG_SIZE_0=16" in nvcc_line:
        return "ORIG-16 (WG=16)"
    if "-DRD_WG_SIZE_0=32" in nvcc_line:
        return "ORIG-32 (WG=32)"
    m = re.search(r"-DLEGO_R(\d+)=1", nvcc_line)
    if m:
        wg = m.group(1)
        return f"LEGO_R{wg}"
    return "UNKNOWN"

# --- Parse ---
records = []
variant = None
current_size = None

re_size = re.compile(r"\bsize\s*=\s*(\d+)\b")
re_median = re.compile(r"\bavg\s*=\s*([0-9]*\.?[0-9]+)")

for line in raw_log.splitlines():
    s = line.strip()
    if not s:
        continue

    if s.startswith("nvcc"):
        variant = detect_variant(s)
        current_size = None
        continue

    m_size = re_size.search(s)
    if m_size:
        current_size = int(m_size.group(1))
        continue

    m_med = re_median.search(s)
    if m_med and variant is not None and current_size is not None:
        median = float(m_med.group(1))
        records.append({"variant": variant, "size": current_size, "median_ms": median})
        current_size = None  # avoid carryover across blocks

# --- DataFrames ---
df = pd.DataFrame(records)
if df.empty:
    raise ValueError("No data parsed. Check the log format and compiler flags (nvcc lines).")

# Pivot medians: rows = sizes, columns = variants
pivot = (
    df.pivot_table(index="size", columns="variant", values="median_ms", aggfunc="first")
      .sort_index()
)

# --- Compute speedups vs baseline (ORIG-16) ---
baseline_col = "ORIG-16 (WG=16)"

if baseline_col not in pivot.columns:
    raise ValueError(f"Baseline '{baseline_col}' not found. Columns parsed: {list(pivot.columns)}")

# Use all variants present (excluding baseline itself for plotting)
plot_cols = [c for c in pivot.columns if c not in (baseline_col, "ORIG-32 (WG=32)")]
# Reorder so LEGO_R16 comes last
if "LEGO_R16" in plot_cols:
    plot_cols = [c for c in plot_cols if c != "LEGO_R16"]

if not plot_cols:
    raise ValueError("Only baseline found. Need at least one other variant to plot speedups.")

# Speedup = time_baseline / time_variant
speedup = pivot.copy()
for c in speedup.columns:
    speedup[c] = pivot[baseline_col] / pivot[c]

# Keep only sizes with complete data across baseline + all plotted variants
cols_for_speedup = [baseline_col] + plot_cols
speedup_complete = speedup[cols_for_speedup].dropna(how="any")
if speedup_complete.empty:
    raise ValueError("No common matrix sizes across baseline and other variants to plot.")

# --- Plot grouped bars of speedups (excluding the baseline column) ---
sizes = speedup_complete.index.tolist()
x = np.arange(len(sizes))

width = 0.85 / max(1, len(plot_cols))  # keep groups reasonably tight
offsets = (np.arange(len(plot_cols)) - (len(plot_cols) - 1) / 2.0) * width

fig, ax = plt.subplots(figsize=(5, 3))
bars = []
col_labels = {"LEGO_R2": "R=2", "LEGO_R4": "R=4", "LEGO_R8": "R=8"}
for i, col in enumerate(plot_cols):
    vals = speedup_complete[col].values
    bar = ax.bar(x + offsets[i], vals, width, label=col_labels.get(col))
    bars.append(bar)

ax.set_xlabel("Matrix size (N)")
ax.set_ylabel("Speedup")
# ax.set_title("LU Decomposition (CUDA) — Speedup (higher is better)")
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in sizes])

# Reference line at 1.0×
ax.axhline(y=1.0, linestyle="--", linewidth=1)

# # Add bar labels
# for grp in bars:
#     for rect in grp:
#         h = rect.get_height()
#         if np.isfinite(h):
#             ax.annotate(f"{h:.2f}x",
#                         (rect.get_x() + rect.get_width() / 2.0, h),
#                         xytext=(0, 3), textcoords="offset points",
#                         ha="center", va="bottom", fontsize=8)

ax.legend(ncols=1)
ax.grid(True, linestyle="--", linewidth=0.5, axis="y")
plt.tight_layout()
plt.savefig("lud.pdf")
# plt.show()