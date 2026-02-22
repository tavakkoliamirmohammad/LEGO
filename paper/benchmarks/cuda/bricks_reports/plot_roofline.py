#!/usr/bin/env python3
import argparse
import glob
import math
import os
import re
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Helpers ----------


def normalize_cols(df):
    m = {c: re.sub(r'\s+', ' ', str(c)).strip().lower() for c in df.columns}
    return df.rename(columns=m)


def to_float(x):
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if s.lower() in ("", "nan", "inf", "infinity", "none"):
        return float('nan')
    # drop thousands separators
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return float('nan')


def read_roofline_csv(path):
    # robustly read comma or tab delimited
    try:
        df = pd.read_csv(path, engine='python')
    except Exception:
        df = pd.read_csv(path, sep=None, engine='python')
    return normalize_cols(df)


def pick_single_precision_rows(df):
    # keep only roofline-related rows (some exports repeat other sections)
    mask_roof = df['section name'].str.contains(
        'roofline', case=False, na=False) if 'section name' in df.columns else pd.Series([True]*len(df))
    df = df[mask_roof]
    # identify SP "Roofline" and "Achieved Value" blocks
    body = df['body item label'] if 'body item label' in df.columns else pd.Series([
                                                                                     '']*len(df))
    sp_roof = df[body.str.contains(
        'single precision roofline', case=False, na=False)]
    sp_achv = df[body.str.contains(
        'single precision achieved value', case=False, na=False)]
    return sp_roof, sp_achv


def extract_peaks(sp_roof):
    # Needed metrics:
    # - Theoretical Predicated-On FFMA Operations (inst)    -> ops per cycle
    # - SM Frequency (Ghz)                                  -> cycles per second
    # - Theoretical DRAM Bytes Accessible (Kbyte/cycle)
    # - DRAM Frequency (Ghz)
    ops = None
    sm_ghz = None
    kbyte_per_cycle = None
    dram_ghz = None
    for _, r in sp_roof.iterrows():
        mname = str(r.get('metric name', '')).strip().lower()
        val = to_float(r.get('metric value', 'nan'))
        unit = str(r.get('metric unit', '')).lower()
        if 'ffma' in mname and 'single' in str(r.get('body item label', '')).lower():
            # "Theoretical Predicated-On FFMA Operations"
            if math.isnan(val):
                continue
            ops = val  # interpret as ops per cycle
        elif mname == 'sm frequency' or ('sm frequency' in mname):
            sm_ghz = val
        elif 'dram bytes accessible' in mname and 'kbyte' in unit:
            kbyte_per_cycle = val
        elif mname == 'dram frequency' or ('dram frequency' in mname):
            dram_ghz = val

    if None in (ops, sm_ghz, kbyte_per_cycle, dram_ghz):
        print(ops, sm_ghz, kbyte_per_cycle, dram_ghz)
        raise RuntimeError(
            "Could not extract peaks from roofline CSV block. Check that Single Precision Roofline rows are present.")
    # FFMA 'operations' already account for 2 flops each
    peak_compute_gflops = ops * sm_ghz
    peak_mem_gbs = kbyte_per_cycle * dram_ghz * \
        1e3  # (KByte/cyc) * (GHz) -> GByte/s
    return peak_compute_gflops, peak_mem_gbs


def extract_point(sp_achv):
    # Achieved:
    # - Predicated-On FFMA Operations Per Cycle (inst)
    # - Predicated-On FADD Thread Instructions Executed Per Cycle (inst/cycle)
    # - Predicated-On FMUL Thread Instructions Executed Per Cycle (inst/cycle)
    # - SM Frequency (Ghz)
    # - DRAM Bandwidth (Gbyte/s)
    ffma_ops_per_cyc = fadd_per_cyc = fmul_per_cyc = sm_ghz = dram_gbs = None
    for _, r in sp_achv.iterrows():
        mname = str(r.get('metric name', '')).strip().lower()
        val = to_float(r.get('metric value', 'nan'))
        unit = str(r.get('metric unit', '')).lower()
        if 'ffma' in mname and 'per cycle' in mname and 'operations' in mname:
            ffma_ops_per_cyc = val
        elif 'fadd' in mname and 'per cycle' in mname:
            fadd_per_cyc = val
        elif 'fmul' in mname and 'per cycle' in mname:
            fmul_per_cyc = val
        elif mname == 'sm frequency' or ('sm frequency' in mname):
            sm_ghz = val
        elif 'dram bandwidth' in mname and ('gbyte/s' in unit or 'gb/s' in unit or 'gbyte/s' in mname):
            dram_gbs = val

    if None in (ffma_ops_per_cyc, fadd_per_cyc, fmul_per_cyc, sm_ghz, dram_gbs):
        raise RuntimeError(
            "Could not extract achieved rows from roofline CSV block.")
    achieved_gflops = (ffma_ops_per_cyc + fadd_per_cyc + fmul_per_cyc) * sm_ghz
    ai_flops_per_byte = achieved_gflops / max(dram_gbs, 1e-12)
    return achieved_gflops, ai_flops_per_byte, dram_gbs


# MODIFIED: Used to capture kind, method, and repetition number
FNPAT = re.compile(r"ncu-run-(cube|star)-(naive|bricks)-r(\d+)\.roofline\.csv$")


def parse_label_from_filename(fname):
    m = FNPAT.search(os.path.basename(fname))
    if not m:
        return None
    kind, method, r_str = m.groups()
    r = int(r_str)
    
    # The original key includes the repetition number 'r' (for filtering/sorting)
    original_key = f"{kind}-{method}-{r}"
    # The new group key excludes 'r' (for grouping the plot)
    series_group_key = f"{kind}-{method}"

    if kind == 'cube':
        label = f"Cube-{method}(R={r})"
    else:
        label = f"Star-{method}(R={r})"
        
    # Return both keys and the label
    return original_key, series_group_key, label

# ---------- Main plotting (Modified) ----------


def main():
    ap = argparse.ArgumentParser(
        description="Plot multiple Nsight Compute roofline CSVs on one chart, connecting all problem sizes per method.")
    ap.add_argument("--dir", type=str, default="./",
                    help="Directory containing *.roofline.csv files.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[
                        2048, 4096, 8192, 16384, 16384*2], help="Problem sizes to include and connect.")
    ap.add_argument("--include", type=str, nargs="+", default=["cube-naive-1", "cube-naive-2", "star-naive-1", "star-naive-2", "star-naive-3", "star-naive-4", "cube-bricks-1", "cube-bricks-2", "star-bricks-1", "star-bricks-2", "star-bricks-3", "star-bricks-4"],
                    help="Series keys to include (derived from filenames). Example: lego-2 lego-4 lego-8 orig-32")
    ap.add_argument("--out", type=str,
                    default="stencil_roofline.png", help="Output PNG path.")
    ap.add_argument("--table-out", type=str, default=None,
                    help="Optional CSV to dump the computed points.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.roofline.csv")))
    if not files:
        raise SystemExit(f"No roofline CSVs found in {args.dir}")

    # Change 'points' to group by the method (e.g., 'cube-naive')
    points_by_method = defaultdict(list) 
    peaks_compute = []
    peaks_mem = []

    for f in files:
        parsed = parse_label_from_filename(f)
        if not parsed:
            continue
            
        original_key, series_group_key, label = parsed
        
        # Use the original key for checking inclusion
        if original_key not in args.include:
            continue

        df = read_roofline_csv(f)
        # Some exports lowercase column names differently; ensure we have expected columns
        for col in ["section name", "body item label", "metric name", "metric unit", "metric value"]:
            if col not in df.columns:
                raise SystemExit(f"{f}: missing expected column '{col}'")

        sp_roof, sp_achv = pick_single_precision_rows(df)
        if sp_roof.empty or sp_achv.empty:
            raise SystemExit(
                f"{f}: could not find Single Precision Roofline/Achieved Value rows")
        print(f)
        peak_gflops, peak_mem_gbs = extract_peaks(sp_roof)
        peaks_compute.append(peak_gflops)
        peaks_mem.append(peak_mem_gbs)

        achieved_gflops, ai, dram_gbs = extract_point(sp_achv)

        # Append to the list for the method group
        points_by_method[series_group_key].append({
            "original_key": original_key, # Keep original key for sorting/table
            "label": label,
            "AI_FLOPs_per_Byte": ai,
            "Perf_GFLOP_s": achieved_gflops,
            "BW_GB_s": dram_gbs,
            "file": os.path.basename(f)
        })

    # Consistency check: use median peaks (should be identical across runs)
    if not peaks_compute or not peaks_mem:
        raise SystemExit("Could not infer device ceilings from the CSVs.")
    peak_compute_gflops = float(np.median(peaks_compute))
    peak_mem_gbs = float(np.median(peaks_mem))

    # Prepare plot data for determining plot range
    x_all, y_all = [], []
    for group_key in points_by_method:
        pts = points_by_method[group_key]
        x_all.extend([p["AI_FLOPs_per_Byte"] for p in pts])
        y_all.extend([p["Perf_GFLOP_s"] for p in pts])
    
    if not x_all:
        raise SystemExit(
            "No matching points after filtering. Check --include and --sizes.")

    # Build roofline curve
    xmin = max(min(x_all)*0.5, 1e-3)
    xmax = max(x_all)*2.0
    xgrid = np.logspace(math.log10(xmin), math.log10(xmax), 256)
    yroof = np.minimum(peak_compute_gflops, peak_mem_gbs * xgrid)
    
    # Plot setup
    plt.figure(figsize=(5, 3))
    plt.loglog(xgrid, yroof, linewidth=1, label="Roofline", color='black')
    # plt.loglog(xgrid, np.full_like(xgrid, peak_compute_gflops),
            #    linestyle='--', linewidth=1)
    # plt.loglog(xgrid, peak_mem_gbs * xgrid, linestyle='--', linewidth=1)

    # Plot data points, grouped by method
    method_groups = sorted(points_by_method.keys())
    marker_cycle = ['o', 's', 'o', 's', 'P', 'X', 'v', '<', '>']
    
    for idx, group_key in enumerate(method_groups):
        pts = points_by_method[group_key]
        
        # CRITICAL: Sort points by the repetition number (last part of original_key)
        # to connect them in the correct order (e.g., r1 -> r2)
        pts_sorted = sorted(pts, key=lambda p: int(p["original_key"].split('-')[-1]))
        
        xs = [p["AI_FLOPs_per_Byte"] for p in pts_sorted]
        ys = [p["Perf_GFLOP_s"] for p in pts_sorted]
        
        # Generate a clean label for the series (e.g., 'Cube Naive')
        series_label = group_key.replace('-', ' ').replace('naive', 'baseline').replace('bricks', 'LEGO').title() 
        
        mk = marker_cycle[idx % len(marker_cycle)]
        plt.loglog(xs, ys, marker=mk, linestyle='-', # linestyle='-' connects the points
                   linewidth=1.5, label=series_label, color="red" if 'Cube' in series_label else "blue")
                   
    plt.xlabel("Arithmetic Intensity (FLOPs / Byte)")
    plt.ylabel("Achieved Performance (GFLOP/s)")
    plt.legend(loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout()

    out = Path(args.out)
    plt.savefig(out, dpi=200)
    print(f"Saved: {out.resolve()}")

    # write summary table if requested
    if args.table_out:
        rows = []
        for group_key in method_groups:
            # Sort points for a consistent table output
            pts = sorted(points_by_method[group_key], key=lambda p: p["original_key"])
            for p in pts:
                rows.append({
                    "Series": p["label"],
                    "AI_FLOPs_per_Byte": p["AI_FLOPs_per_Byte"],
                    "Perf_GFLOP_s": p["Perf_GFLOP_s"],
                    "BW_GB_s": p["BW_GB_s"],
                    "File": p["file"]
                })
        df = pd.DataFrame(rows)
        df = df.sort_values(by=["Series"])
        df.to_csv(args.table_out, index=False)
        print(f"Wrote table: {Path(args.table_out).resolve()}")


if __name__ == "__main__":
    main()