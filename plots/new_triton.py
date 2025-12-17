#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_text() -> str:
    prefix = "../result/triton/"
    path = ["group_gemm.txt", "softmax.txt",
            "layernorm.txt", "softmax.txt", "matmul.txt"]
    res = ""
    for p in path:
        with open(prefix + p, 'r', encoding='utf-8') as f:
            res += f.read() + "\n\n"
    return res


def parse_tables(text: str) -> Dict[str, pd.DataFrame]:
    blocks: Dict[str, str] = {}
    label_re = re.compile(
        r'(?P<label>^[A-Za-z0-9\-\s\^\u1D40\u1D3F\u1D2C\u1D2F\u00B2\^\u1D43\u1D47]+:)\n', re.MULTILINE)
    matches = list(label_re.finditer(text))

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        label = m.group('label').strip(':').strip().lower()
        block = text[start:end].strip()
        if re.search(r'\n', block) and re.search(r'[A-Za-z]', block.splitlines()[0]):
            blocks[label] = block

    dfs: Dict[str, pd.DataFrame] = {}
    for label, block in blocks.items():
        try:
            df = pd.read_table(pd.io.common.StringIO(block), sep=r'\s+', engine='python')
            dfs[label] = df
        except Exception:
            clean = re.sub(r'[,\|]+', ' ', block)
            try:
                df = pd.read_table(pd.io.common.StringIO(clean), sep=r'\s+', engine='python')
                dfs[label] = df
            except Exception:
                pass
    return dfs


def in_range(val, lo=512, hi=8192):
    try:
        v = float(val)
        return (v >= lo) and (v <= hi)
    except Exception:
        return False


def speedup_series(xs: List[float], base: List[float], triton: List[float], lego: List[float]):
    # kept for compatibility, not used in the Triton-as-baseline line charts
    t = [t/b if b != 0 else float('nan') for t, b in zip(triton, base)]
    l = [l/b if b != 0 else float('nan') for l, b in zip(lego, base)]
    return t, l


def bar_chart(xs, t_speed, l_speed, title, baseline_label, out_pdf):
    x = np.arange(len(xs))
    width = 0.38
    plt.figure(figsize=(7, 4), dpi=150)
    plt.bar(x - width/2, t_speed, width, label='Triton')
    plt.bar(x + width/2, l_speed, width, label='LEGO')
    plt.axhline(1.0, linestyle='--', linewidth=1, color='k')
    plt.xticks(x, [str(int(xi)) for xi in xs])
    plt.xlabel('Problem size (N)')
    plt.ylabel('Speedup')
    plt.ylim(bottom=0)
    plt.grid(True, axis='y', linewidth=0.3)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()


def grouped_matmul_2x2(fig_title, grids, out_pdf):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    for ax, (ptitle, xs, t_speed, l_speed) in zip(axes.flat, grids):
        x = np.arange(len(xs))
        width = 0.38
        ax.bar(x - width/2, t_speed, width, label='Triton')
        ax.bar(x + width/2, l_speed, width, label='LEGO')
        ax.axhline(1.0, linestyle='--', linewidth=1, color='k')
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(xi)) for xi in xs])
        ax.set_title(ptitle)
        ax.set_ylabel('Speedup')
        ax.set_ylim(bottom=0)
        ax.grid(True, axis='y', linewidth=0.3)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()


def grouped_layernorm_1x2(df_fwd: pd.DataFrame, df_bwd: pd.DataFrame, out_pdf: str):
    def prep(df):
        mask = df['n'].apply(in_range)
        dff = df[mask].copy()
        xs = dff['n'].astype(float).tolist()
        base = dff['torch'].astype(float).tolist()
        t = dff['triton'].astype(float).tolist()
        l = dff['lego'].astype(float).tolist()
        t_speed, l_speed = speedup_series(xs, base, t, l)
        return xs, t_speed, l_speed

    xs_f, t_f, l_f = prep(df_fwd)
    xs_b, t_b, l_b = prep(df_bwd)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    ax = axes[0]
    x = np.arange(len(xs_f)); width = 0.38
    ax.bar(x - width/2, t_f, width, label='Triton')
    ax.bar(x + width/2, l_f, width, label='LEGO')
    ax.axhline(1.0, linestyle='--', linewidth=1, color='k')
    ax.set_xticks(x); ax.set_xticklabels([str(int(xi)) for xi in xs_f])
    ax.set_title('LayerNorm Forward'); ax.set_ylim(bottom=0)
    ax.grid(True, axis='y', linewidth=0.3)
    ax.set_ylabel('Speedup')

    ax = axes[1]
    x = np.arange(len(xs_b)); width = 0.38
    ax.bar(x - width/2, t_b, width, label='Triton')
    ax.bar(x + width/2, l_b, width, label='LEGO')
    ax.axhline(1.0, linestyle='--', linewidth=1, color='k')
    ax.set_xticks(x); ax.set_xticklabels([str(int(xi)) for xi in xs_b])
    ax.set_title('LayerNorm Backward'); ax.set_ylim(bottom=0)
    ax.grid(True, axis='y', linewidth=0.3)
    ax.set_ylabel('Speedup')

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()


def build_matmul_sets(dfs: Dict[str, pd.DataFrame]):
    variants = ['a b', 'a^t b', 'a b^t', 'a^t b^t']
    fp16 = {}; fp8 = {}
    for var in variants:
        key16 = None
        for k in list(dfs.keys()):
            if 'matmul-performance-fp16' in k and var in k:
                key16 = k; break
        if key16 is None:
            candidates = [k for k in dfs.keys() if 'matmul-performance-fp16' in k]
            if len(candidates) == 1: key16 = candidates[0]

        key8 = None
        for k in list(dfs.keys()):
            if 'matmul-performance-fp8' in k and var in k:
                key8 = k; break
        if key8 is None:
            candidates = [k for k in dfs.keys() if 'matmul-performance-fp8' in k]
            if len(candidates) == 1: key8 = candidates[0]

        if key16: fp16[var] = dfs[key16].copy()
        if key8:  fp8[var]  = dfs[key8].copy()
    return fp16, fp8


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def extract_mm(df: pd.DataFrame, baseline_col='cublas'):
    df = normalize_columns(df)
    xs = df['n'].astype(float).tolist() if 'n' in df.columns else df.iloc[:,1].astype(float).tolist()
    mask = [in_range(x) for x in xs]
    def filt(col):
        return [float(v) for v, m in zip(df[col].tolist(), mask) if m]
    xs = [x for x, m in zip(xs, mask) if m]
    base = filt(baseline_col) if (baseline_col and baseline_col in df.columns) else None
    t = filt('triton'); l = filt('lego')
    return xs, base, t, l


def make_all_benchmarks_boxplot(speedup_map: Dict[str, Tuple[List[float], List[float]]], out_pdf: str):
    labels = list(speedup_map.keys())
    lego_data = [speedup_map[k][0] for k in labels]
    triton_data = [speedup_map[k][1] for k in labels]

    n = len(labels)
    base_positions = np.arange(n) * 3.0
    pos_lego = base_positions - 0.5
    pos_trit = base_positions + 0.5

    plt.figure(figsize=(max(10, n*1.6), 6), dpi=150)

    bp1 = plt.boxplot(
        lego_data, positions=pos_lego, widths=0.9, patch_artist=True, showfliers=False, whis=[0, 100]
    )
    bp2 = plt.boxplot(
        triton_data, positions=pos_trit, widths=0.9, patch_artist=True, showfliers=False, whis=[0, 100]
    )

    for patch in bp1['boxes']:
        patch.set_facecolor('orange')
    for patch in bp2['boxes']:
        patch.set_facecolor('royalblue')

    plt.axhline(1.0, linestyle='--', linewidth=1, color='k')
    tick_positions = base_positions
    plt.xticks(tick_positions, labels, rotation=20, ha='right')
    plt.ylabel('Speedup')
    plt.ylim(bottom=0)
    plt.grid(True, axis='y', linewidth=0.3)

    from matplotlib.patches import Patch
    lego_patch = Patch(facecolor=bp1['boxes'][0].get_facecolor(), label='LEGO')
    trit_patch = Patch(facecolor=bp2['boxes'][0].get_facecolor(), label='Triton')
    plt.legend(handles=[lego_patch, trit_patch], loc='upper left', ncol=2)

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()


# ---------------------- NEW / TRITON-BASELINE PARTS ----------------------

def pick_vs_triton_at(df: pd.DataFrame, N: int) -> Optional[Tuple[float, float]]:
    """
    Return (lego_vs_triton, ref_vs_triton) at specific N, where:
      - Triton is the baseline (1×)
      - ref is PyTorch if present, otherwise cuBLAS if present
    """
    df = normalize_columns(df)
    if 'n' not in df.columns or 'triton' not in df.columns:
        return None
    row = df[df['n'].astype(float) == float(N)]
    if row.empty:
        return None
    triton = float(row['triton'].iloc[0])
    if triton is None or triton == 0:
        return None
    lego = float(row['lego'].iloc[0]) if 'lego' in df.columns else None
    ref_col = 'torch' if 'torch' in df.columns else ('cublas' if 'cublas' in df.columns else None)
    ref = float(row[ref_col].iloc[0]) if ref_col and ref_col in df.columns else None
    if lego is None or ref is None:
        return None
    return (lego / triton), (ref / triton)


def collect_speedups_at_N(dfs: Dict[str, pd.DataFrame], N: int) -> List[Tuple[str, float, float]]:
    """Build [(label, lego_vs_triton, ref_vs_triton)] for a given N."""
    items: List[Tuple[str, float, float]] = []

    # Group GEMM (ref = cuBLAS)
    for key in list(dfs.keys()):
        if 'group-gemm-performance' in key:
            s = pick_vs_triton_at(dfs[key], N=N)
            if s:
                lego_vs_triton, ref_vs_triton = s
                items.append(('Group GEMM', lego_vs_triton, ref_vs_triton))
            break

    # LayerNorm Fwd/Bwd (ref = PyTorch)
    for key in list(dfs.keys()):
        if 'layer-norm-forward' in key:
            s = pick_vs_triton_at(dfs[key], N=N)
            if s:
                lego_vs_triton, ref_vs_triton = s
                items.append(('LayerNorm Fwd', lego_vs_triton, ref_vs_triton))
            break

    for key in list(dfs.keys()):
        if 'layer-norm-backward' in key:
            s = pick_vs_triton_at(dfs[key], N=N)
            if s:
                lego_vs_triton, ref_vs_triton = s
                items.append(('LayerNorm Bwd', lego_vs_triton, ref_vs_triton))
            break

    # Softmax (ref = PyTorch)
    for key in list(dfs.keys()):
        if 'softmax-performance' in key:
            s = pick_vs_triton_at(dfs[key], N=N)
            if s:
                lego_vs_triton, ref_vs_triton = s
                items.append(('Softmax', lego_vs_triton, ref_vs_triton))
            break

    # Matmul fp16 (ref = cuBLAS)
    fp16_sets, _ = build_matmul_sets(dfs)
    matmul_map = [
        ('Matmul A B (fp16)', 'a b'),
        ('Matmul A^T B (fp16)', 'a^t b'),
        ('Matmul A B^T (fp16)', 'a b^t'),
        ('Matmul A^T B^T (fp16)', 'a^t b^t'),
    ]
    for label, key in matmul_map:
        if key in fp16_sets:
            s = pick_vs_triton_at(fp16_sets[key], N=N)
            if s:
                lego_vs_triton, ref_vs_triton = s
                items.append((label, lego_vs_triton, ref_vs_triton))

    # Filter any incomplete entries
    items = [(lbl, l_trit, ref_trit) for (lbl, l_trit, ref_trit) in items
             if (l_trit is not None and ref_trit is not None)]
    return items


def line_chart_at_N(speedup_items: List[Tuple[str, float, float]], N: int, out_pdf: str):
    """
    speedup_items: list of (label, lego_vs_triton, ref_vs_triton) at given N
    Triton is the implicit baseline (1×), drawn as a horizontal line.
    """
    labels = [lbl for lbl, _, _ in speedup_items]
    lego_vals = [l for _, l, _ in speedup_items]
    ref_vals  = [r for _, _, r in speedup_items]

    x = np.arange(len(labels))
    plt.figure(figsize=(6, 3.6), dpi=150)
    # Markers only; Triton shown as reference line
    plt.scatter(x, lego_vals, marker='x', label='LEGO')
    plt.scatter(x, ref_vals,  marker='o', label='PyTorch/cuBLAS')
    plt.axhline(1.0, linestyle='--', linewidth=1, color='k', label='Triton')
    plt.xticks(x, labels, rotation=20, ha='right')
    plt.ylabel('Speedup')
    plt.ylim(bottom=0, top=1.5)
    plt.grid(True, axis='y', linewidth=0.3)
    plt.legend(loc='lower right')
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', dest='outdir', help='Output directory', default="./charts")
    ap.add_argument('--sizes', type=str, default="2048,4096,8192",
                    help="Comma-separated problem sizes to plot (e.g., 1024,2048,8192)")
    args = ap.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(',') if s.strip()]

    text = read_text()
    dfs = parse_tables(text)
    for k in list(dfs.keys()):
        dfs[k] = normalize_columns(dfs[k])

    for N in sizes:
        speedup_items = collect_speedups_at_N(dfs, N)
        if not speedup_items:
            # nothing to plot at this N (skip silently)
            continue
        out_pdf = os.path.join(args.outdir, f'speedup_at_{N}_linechart.pdf')
        line_chart_at_N(speedup_items, N, out_pdf)


if __name__ == '__main__':
    main()
