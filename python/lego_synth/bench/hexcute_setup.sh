#!/usr/bin/env bash
# Build the Hexcute baseline (the CGO'26 artifact's hidet fork) in its own Python environment.
#   HEXCUTE_WS   where to clone/build (default: <repo>/../hexcute-bench/workspace)
#   PYTHON       base interpreter (>= 3.10) used to create the venv
#   CMAKE        cmake >= 3.19 (default: the one on PATH)
# Result: $HEXCUTE_WS/hidet (source + build/lib/libhidet*.so) and $HEXCUTE_WS/hexcute-venv with torch + hidet.
# The runtime library is built with the C++ compiler on PATH; run the benchmark with the matching
# libstdc++ on LD_LIBRARY_PATH (scripts/env.sh does this when CC points at a non-system gcc).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
WS=${HEXCUTE_WS:-$ROOT/../hexcute-bench/workspace}
PYTHON=${PYTHON:-python3}
CMAKE=${CMAKE:-cmake}
mkdir -p "$WS"
if [ ! -d "$WS/hidet/.git" ]; then
  git clone --depth 1 -b cgorebuttal https://github.com/hexcute/hidet.git "$WS/hidet"
fi
if [ ! -f "$WS/hidet/build/lib/libhidet_runtime.so" ]; then
  mkdir -p "$WS/hidet/build" && cd "$WS/hidet/build" && "$CMAKE" .. -DCMAKE_BUILD_TYPE=Release && make -j"${JOBS:-8}"
fi
if [ ! -x "$WS/hexcute-venv/bin/python" ]; then
  "$PYTHON" -m venv "$WS/hexcute-venv"
  "$WS/hexcute-venv/bin/pip" install -q -U pip
  # torch (with its CUDA runtime) plus hidet's own dependencies; hip-python-fork is not needed on NVIDIA
  "$WS/hexcute-venv/bin/pip" install -q torch numpy psutil tabulate tqdm nvtx click packaging requests filelock "cuda-python>=11.6.1" lark tomlkit scipy gitpython importlib_metadata networkx
  "$WS/hexcute-venv/bin/pip" install -q --no-deps -e "$WS/hidet"
fi
"$WS/hexcute-venv/bin/python" -c "import hidet, torch; print('hexcute env ok: hidet', hidet.__version__, 'torch', torch.__version__)"
echo "run: HEXCUTE_ROOT=$WS/hidet $WS/hexcute-venv/bin/python $HERE/hexcute_attention.py --seq 1024 --dims 64"
