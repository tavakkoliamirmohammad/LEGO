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
    # - Theoretical Predicated-On FFMA Operations (inst)   -> ops per cycle
    # - SM Frequency (Ghz)                                 -> cycles per second
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
    print(sp_achv)
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


FNPAT = re.compile(r"ncu-run-(original|antidiag)-k(\d+)-s(\d+)\.roofline\.csv$")


def parse_label_from_filename(fname):
    m = FNPAT.search(os.path.basename(fname))
    if not m:
        return None
    kind, k_str, size_str = m.groups()
    size = int(size_str)
    if kind == 'antidiag':
        label = f"antidiag(k={k_str})"
        key = f"antidiag-k{k_str}"
    else:
        label = f"Original(k={k_str})"
        key = f"original-k{k_str}"
    return key, label, size

# ---------- Main plotting ----------


def main():
    ap = argparse.ArgumentParser(
        description="Plot multiple Nsight Compute roofline CSVs on one chart, connecting problem sizes per method.")
    ap.add_argument("--dir", type=str, default="reports",
                    help="Directory containing *.roofline.csv files.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[
                    1024, 2048, 4096, 8192, 16384, 16384*2], help="Problem sizes to include and connect.")
    ap.add_argument("--include", type=str, nargs="+", default=["original-k1", "original-k2", "antidiag-k1", "antidiag-k2"],
                    help="Series keys to include (derived from filenames). Example: lego-2 lego-4 lego-8 orig-32")
    ap.add_argument("--out", type=str,
                    default="multi_roofline.png", help="Output PNG path.")
    ap.add_argument("--table-out", type=str, default=None,
                    help="Optional CSV to dump the computed points.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.roofline.csv")))
    if not files:
        raise SystemExit(f"No roofline CSVs found in {args.dir}")

    points = defaultdict(list)
    peaks_compute = []
    peaks_mem = []

    for f in files:
        parsed = parse_label_from_filename(f)
        if not parsed:
            continue
        key, label, size = parsed
        if key not in args.include or size not in set(args.sizes):
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

        points[key].append({
            "label": label, "size": size,
            "AI_FLOPs_per_Byte": ai,
            "Perf_GFLOP_s": achieved_gflops,
            "BW_GB_s": dram_gbs,
            "file": os.path.basename(f)
        })
        print(points[key][-1])

    # Consistency check: use median peaks (should be identical across runs)
    if not peaks_compute or not peaks_mem:
        raise SystemExit("Could not infer device ceilings from the CSVs.")
    peak_compute_gflops = float(np.median(peaks_compute))
    peak_mem_gbs = float(np.median(peaks_mem))

    # Prepare plot data, ensuring sizes are in ascending order per series
    x_all, y_all = [], []
    series_order = []
    for key in args.include:
        if key not in points:
            continue
        series_order.append(key)
        pts = sorted(points[key], key=lambda d: d["size"])
        x_all.extend([p["AI_FLOPs_per_Byte"] for p in pts])
        y_all.extend([p["Perf_GFLOP_s"] for p in pts])

    if not x_all:
        raise SystemExit(
            "No matching points after filtering. Check --include and --sizes.")

    # Build roofline curve
    print(x_all)
    xmin = max(min(x_all)*0.5, 1e-3)
    xmax = max(x_all)*2.0
    xgrid = np.logspace(math.log10(xmin), math.log10(xmax), 256)
    yroof = np.minimum(peak_compute_gflops, peak_mem_gbs * xgrid)
    # Plot
    plt.figure(figsize=(9, 6))
    plt.loglog(xgrid, yroof, linewidth=2, label="Roofline")
    plt.loglog(xgrid, np.full_like(xgrid, peak_compute_gflops),
               linestyle='--', linewidth=1)
    plt.loglog(xgrid, peak_mem_gbs * xgrid, linestyle='--', linewidth=1)

    marker_cycle = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>']
    for idx, key in enumerate(series_order):
        pts = sorted(points[key], key=lambda d: d["size"])
        xs = [p["AI_FLOPs_per_Byte"] for p in pts]
        ys = [p["Perf_GFLOP_s"] for p in pts]
        label = pts[0]["label"] if pts else key
        mk = marker_cycle[idx % len(marker_cycle)]
        plt.loglog(xs, ys, marker=mk, linestyle='-',
                   linewidth=1.5, label=label)
        # annotate sizes
        for p in pts:
            sz = p["size"]
            txt = f"{sz//1024 if sz % 1024 == 0 else sz}k" if sz >= 1024 else f"{sz}"
            # slightly offset text to avoid overlap

    plt.title("GPU Roofline(LUD)")
    plt.xlabel("Arithmetic Intensity (FLOPs / Byte)")
    plt.ylabel("Achieved Performance (GFLOP/s)")
    plt.grid(True, which='both', linestyle=':')
    plt.legend(loc='lower right')
    plt.tight_layout()

    out = Path(args.out)
    plt.savefig(out, dpi=200)
    print(f"Saved: {out.resolve()}")

    # write summary table if requested
    if args.table_out:
        rows = []
        for key in series_order:
            for p in points[key]:
                rows.append({
                    "Series": p["label"],
                    "Size": p["size"],
                    "AI_FLOPs_per_Byte": p["AI_FLOPs_per_Byte"],
                    "Perf_GFLOP_s": p["Perf_GFLOP_s"],
                    "BW_GB_s": p["BW_GB_s"],
                    "File": p["file"]
                })
        df = pd.DataFrame(rows)
        df = df.sort_values(by=["Series", "Size"])
        df.to_csv(args.table_out, index=False)
        print(f"Wrote table: {Path(args.table_out).resolve()}")


if __name__ == "__main__":
    main()
