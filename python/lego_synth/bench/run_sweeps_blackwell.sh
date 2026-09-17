#!/usr/bin/env bash
# The GPU-timed part of run_all.sh (GEMM, attention x2, Hexcute, cuBLAS probe); the CPU ablations and
# calibrations already ran. Started detached; log: results/run_sweeps.log
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python3}"
echo "== GEMM"; $PY -m lego_synth.bench.gemm_sweep --sizes 1024,2048,4096 --shapes "4096x6144x4096;4096x14336x4096;4096x4096x14336;4096x11008x4096;4096x4096x11008;4096x18944x3584;4096x3584x18944"
echo "== attention non-causal"; $PY -m lego_synth.bench.attn_sweep --bh 32 --seq 1024,2048,4096 --dims 64,80,96,112,128
echo "== attention causal"; $PY -m lego_synth.bench.attn_sweep --bh 32 --seq 1024,2048,4096 --dims 64,80,96,112,128 --causal
HEX_WS="${HEXCUTE_WS:-$(pwd)/../hexcute-bench/workspace}"
if [ -x "$HEX_WS/hexcute-venv/bin/python" ]; then
  echo "== Hexcute"; HEXCUTE_ROOT="$HEX_WS/hidet" "$HEX_WS/hexcute-venv/bin/python" lego_synth/bench/hexcute_attention.py --bh 32 --seq 1024,2048,4096 --dims 64,80,96,112,128 || echo "Hexcute run failed"
  echo "== cuBLAS in the Hexcute env (second CUDA version)"; "$HEX_WS/hexcute-venv/bin/python" lego_synth/bench/cublas_probe.py || echo "cuBLAS probe failed"
fi
echo "== done"
