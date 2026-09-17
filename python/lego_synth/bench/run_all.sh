#!/usr/bin/env bash
# Run the complete CASTLE benchmark suite on the GPU in this machine. Results land in
# python/lego_synth/results/<device-tag>/ with a provenance header in every file.
#   source scripts/env.sh && bash python/lego_synth/bench/run_all.sh [quick]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"
QUICK="${1:-}"
echo "== device"; $PY -c "from lego_synth.kernels.target import detect_target as d; print(d())"
echo "== solver ablations (CPU)"
$PY -m lego_synth.scripts.ablation --kind staging --elem 2 --rows 64 --widths 32,48,64,80,96,112,128 --out lego_synth/results/ablation_staging_b16.md
$PY -m lego_synth.scripts.ablation --kind transpose --elem 4 --rows 32 --out lego_synth/results/ablation_f32.md
echo "== hardware calibration (GPU)"
$PY -m lego_synth.scripts.calibrate_smem
$PY -m lego_synth.scripts.bench_ldmatrix_w80
echo "== GEMM layout ablation (GPU)"
$PY -m lego_synth.bench.gemm_layout_ablation
if command -v ncu >/dev/null 2>&1; then
  echo "== bank conflicts: Nsight Compute counters vs model"
  $PY -m lego_synth.scripts.ncu_conflicts || echo "ncu counters unavailable on this node; skipping (timing calibration above stands in)"
fi
if [ "$QUICK" = "quick" ]; then
  echo "== GEMM (quick)"; $PY -m lego_synth.bench.gemm_sweep --sizes 2048,4096 --shapes "" --configs 128x128x32w2x2,128x128x32w4x2 --stages 3 --group_m 8
  echo "== attention (quick)"; $PY -m lego_synth.bench.attn_sweep --seq 2048 --dims 64,80,128 --configs 64x32s2,128x64s2r32
else
  echo "== GEMM (squares + LLM shapes)"
  $PY -m lego_synth.bench.gemm_sweep --sizes 1024,2048,4096 \
     --shapes "4096x6144x4096;4096x14336x4096;4096x4096x14336;4096x11008x4096;4096x4096x11008;4096x18944x3584;4096x3584x18944"
  echo "== attention, non-causal and causal (baselines: SDPA flash/cuDNN/efficient, Triton fixed + autotuned)"
  $PY -m lego_synth.bench.attn_sweep --bh 32 --seq 1024,2048,4096 --dims 64,80,96,112,128
  $PY -m lego_synth.bench.attn_sweep --bh 32 --seq 1024,2048,4096 --dims 64,80,96,112,128 --causal
fi
# Optional: the Hexcute baseline runs in its own environment (bench/hexcute_setup.sh builds it).
HEX_WS="${HEXCUTE_WS:-$(pwd)/../../hexcute-bench/workspace}"
if [ -x "$HEX_WS/hexcute-venv/bin/python" ] && [ "$QUICK" != "quick" ]; then
  echo "== Hexcute attention baseline"
  HEXCUTE_ROOT="$HEX_WS/hidet" "$HEX_WS/hexcute-venv/bin/python" lego_synth/bench/hexcute_attention.py --bh 32 --seq 1024,2048,4096 --dims 64,80,96,112,128 || echo "Hexcute run failed (see above); continuing"
else
  echo "== Hexcute baseline skipped (no $HEX_WS/hexcute-venv; run bench/hexcute_setup.sh to build it)"
fi
echo "== done: $(ls lego_synth/results/*/ | head -20)"
