#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

LOG_SEP = "----"

def parse_log(text: str) -> List[Dict[str, Optional[float]]]:
    """
    Parse a CUDA log into rows of:
      {'radius': int, 'naive': float|None, 'bricks': float|None}
    Supports blocks separated by "----".
    Works for both:
      f3d_naive/f3d_bricks
      laplacian_naive/laplacian_bricks
      with lines like: "... total (X ms / launch)"
    """
    blocks = [b.strip() for b in text.strip().split(LOG_SEP) if b.strip()]

    re_radius_dims = re.compile(r"\bradius\s*=\s*(\d+)\b", re.IGNORECASE)
    re_radius_nvcc = re.compile(r"-DRD_RADIUS\s*=\s*(\d+)", re.IGNORECASE)

    per_launch = r"\(([\d\.]+)\s*ms\s*/\s*launch\)"
    re_naive  = re.compile(r"(?:f3d|laplacian)_naive\s*:\s*[\d\.]+\s*ms\s*total\s*" + per_launch, re.I)
    re_bricks = re.compile(r"(?:f3d|laplacian)_bricks\s*:\s*[\d\.]+\s*ms\s*total\s*" + per_launch, re.I)

    rows: List[Dict[str, Optional[float]]] = []

    for block in blocks:
        mrad = re_radius_dims.search(block) or re_radius_nvcc.search(block)
        if not mrad:
            continue
        radius = int(mrad.group(1))

        n = re_naive.search(block)
        b = re_bricks.search(block)

        naive  = float(n.group(1)) if n else None
        bricks = float(b.group(1)) if b else None

        if naive is not None or bricks is not None:
            rows.append({"radius": radius, "naive": naive, "bricks": bricks})

    rows.sort(key=lambda d: d["radius"])
    return rows

def speedup_by_radius(rows: List[Dict[str, Optional[float]]]) -> Dict[int, float]:
    """Return dict radius -> (bricks/naive) speedup."""
    sp = {}
    for r in rows:
        n, b = r["naive"], r["bricks"]
        if n is not None and b is not None and n > 0 and b > 0:
            sp[r["radius"]] = n/b   # bricks over naive
    return sp
def plot_grouped_speedup_bars(cube_sp: Dict[int, float],
                              lap_sp: Dict[int, float],
                              outpath: Path) -> None:
    # Prepare data
    cube_radii = sorted(cube_sp.keys())
    lap_radii  = sorted(lap_sp.keys())
    y_cube = [cube_sp[r] for r in cube_radii]
    y_lap  = [lap_sp[r]  for r in lap_radii]

    # X positions: two clusters with a small gap between them
    # (smaller gap => clusters closer together)
    gap = 0.5
    x_cube = np.arange(len(cube_radii), dtype=float)
    x_lap  = np.arange(len(lap_radii), dtype=float) + (len(cube_radii) + gap)

    # Plot
    plt.figure(figsize=(5, 3))
    bar_width = 0.95  # make Cube bars sit close to each other
    plt.bar(x_cube, y_cube, width=bar_width, label="Cube")
    plt.bar(x_lap,  y_lap,  width=bar_width, label="Star")

    # Two-line x tick labels: top line R=..., bottom line group name
    xticks = list(x_cube) + list(x_lap)
    xticklabels = ([f"R={r}\nCube" for r in cube_radii] +
                   [f"R={r}\nStar" for r in lap_radii])
    plt.xticks(xticks, xticklabels)
    plt.tick_params(axis="x", pad=6)  # add a bit of padding for the two-line labels

    # Axes and grid
    plt.ylabel("Speedup")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.grid(True, axis="y", linewidth=0.3)
    plt.margins(x=0.02)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser(
        description="Grouped bar chart: Cube (all radii) and Laplace (all radii) speedups."
    )
    ap.add_argument("--out", default="bricks.pdf",
                    help="Output figure path.")
    args = ap.parse_args()

    cube_path = Path("../result/cuda/bricks-f3d.txt")
    lap_path  = Path("../result/cuda/bricks-laplace.txt")
    if not cube_path.exists():
        sys.exit(f"Cube log not found: {cube_path}")
    if not lap_path.exists():
        sys.exit(f"Laplace log not found: {lap_path}")

    cube_rows = parse_log(cube_path.read_text(encoding="utf-8", errors="ignore"))
    lap_rows  = parse_log(lap_path.read_text(encoding="utf-8", errors="ignore"))
    if not cube_rows:
        sys.exit("No Cube data parsed. Check the Cube log format.")
    if not lap_rows:
        sys.exit("No Laplace data parsed. Check the Laplace log format.")

    cube_sp = speedup_by_radius(cube_rows)
    lap_sp  = speedup_by_radius(lap_rows)

    # Console table (optional)
    print("Speedup (bricks/naive):")
    for r in sorted(set(cube_sp.keys()) | set(lap_sp.keys())):
        cs = cube_sp.get(r)
        ls = lap_sp.get(r)
        print(f"  radius={r}:  Cube={cs!s:>8}   Laplace={ls!s:>8}")

    plot_grouped_speedup_bars(cube_sp, lap_sp, Path(args.out))
    print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()
