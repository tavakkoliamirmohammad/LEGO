"""Transpose harness: one CUDA kernel per layout variant, timed with events and profiled with ncu.

The kernel stages a `tile x tile` block through shared memory. Thread (tx, ty)
stores rows r = ty, ty+block_rows, ... with lane -> column (the `row_vector`
pattern) and loads with lane -> row (the `col_vector` pattern), so the model's
prediction for a block is exactly conflicts(L, row_vector) + conflicts(L, col_vector).
"""
import csv
import dataclasses
import io
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, Tuple

from ..eval_np import conflicts
from ..layouts import RowMajor, XorSwizzle, Rotation
from ..machine import ampere
from ..patterns import row_vector, col_vector
from .cuda_emit import index_expr

_FAMILIES = {c.__name__: c for c in (RowMajor, XorSwizzle, Rotation)}
LD_METRIC = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
ST_METRIC = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"


def layout_to_dict(L) -> dict:
    return {"type": type(L).__name__, **dataclasses.asdict(L)}


def layout_from_dict(d: dict):
    d = dict(d)
    return _FAMILIES[d.pop("type")](**d)


def kernel_source(name: str, layout, tile: int, block_rows: int) -> str:
    idx = index_expr(layout)
    return f"""
__global__ void transpose_{name}(const float* __restrict__ in, float* __restrict__ out, int n) {{
  __shared__ float tile[{layout.footprint}];
  const int bx = blockIdx.x * {tile}, by = blockIdx.y * {tile};
  const int tx = threadIdx.x, ty = threadIdx.y;
  #pragma unroll
  for (int r = ty; r < {tile}; r += {block_rows}) {{
    const unsigned i = r, j = tx;
    tile[{idx}] = in[(by + r) * n + bx + tx];
  }}
  __syncthreads();
  #pragma unroll
  for (int r = ty; r < {tile}; r += {block_rows}) {{
    const unsigned i = tx, j = r;
    out[(bx + r) * n + by + tx] = tile[{idx}];
  }}
}}

torch::Tensor run_{name}(torch::Tensor x) {{
  TORCH_CHECK(x.dim() == 2 && x.size(0) == x.size(1) && x.size(0) % {tile} == 0, "square, multiple of tile");
  auto out = torch::empty_like(x);
  const int n = x.size(0);
  dim3 grid(n / {tile}, n / {tile}), block({tile}, {block_rows});
  transpose_{name}<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}}
"""


def predicted_conflicts(layout, n: int, tile: int = 32, machine=ampere()) -> Tuple[int, int]:
    """(load, store) shared-memory conflicts the model predicts for an n x n transpose."""
    assert tile == machine.warp, "one warp per tile row is assumed"
    blocks = (n // tile) ** 2
    st = conflicts(layout, row_vector(tile, tile, machine.warp), 4, machine)
    ld = conflicts(layout, col_vector(tile, tile, machine.warp), 4, machine)
    return ld * blocks, st * blocks


class TransposeHarness:
    def __init__(self, layouts: Dict[str, object], tile: int = 32, block_rows: int = 8):
        self.layouts = dict(layouts)
        self.tile, self.block_rows = tile, block_rows
        self.module = None

    def build(self):
        from torch.utils.cpp_extension import load_inline
        cuda_src = "\n".join(kernel_source(n, L, self.tile, self.block_rows) for n, L in self.layouts.items())
        decls = "\n".join(f"torch::Tensor run_{n}(torch::Tensor x);" for n in self.layouts)
        key = abs(hash((cuda_src, self.tile, self.block_rows))) % (1 << 32)
        self.module = load_inline(name=f"lego_transpose_{key:08x}", cpp_sources=decls, cuda_sources=cuda_src,
                                  functions=[f"run_{n}" for n in self.layouts], verbose=False)
        return self

    def run(self, name: str, x):
        return getattr(self.module, f"run_{name}")(x)

    def time(self, name: str, x, iters: int = 100, warmup: int = 10) -> float:
        import torch
        fn = getattr(self.module, f"run_{name}")
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(x); e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        return times[len(times) // 2]

    def ncu_shared_conflicts(self, n: int) -> Dict[str, Tuple[int, int]]:
        """Run every variant once under Nsight Compute in a child process; return {name: (ld, st)}."""
        ncu = shutil.which("ncu")
        if ncu is None:
            raise RuntimeError("ncu not on PATH")
        spec = {"tile": self.tile, "block_rows": self.block_rows, "n": n,
                "layouts": {k: layout_to_dict(v) for k, v in self.layouts.items()}}
        cmd = [ncu, "--csv", "--metrics", f"{LD_METRIC},{ST_METRIC}", "-k", "regex:transpose_",
               sys.executable, "-m", "lego_synth.harness.transpose_cuda", "--spec", json.dumps(spec)]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        return _parse_ncu_csv(out, self.layouts)


def _parse_ncu_csv(text: str, layouts) -> Dict[str, Tuple[int, int]]:
    lines = [l for l in text.splitlines() if l.startswith('"')]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    acc = {name: [0, 0] for name in layouts}
    for r in rows:
        kname = r.get("Kernel Name", "")
        for name in layouts:
            if kname.startswith(f"transpose_{name}("):
                val = int(float(r["Metric Value"].replace(",", "")))
                if r["Metric Name"] == LD_METRIC:
                    acc[name][0] += val
                elif r["Metric Name"] == ST_METRIC:
                    acc[name][1] += val
    return {k: (v[0], v[1]) for k, v in acc.items()}


def _main(argv):
    import torch
    spec = json.loads(argv[argv.index("--spec") + 1])
    layouts = {k: layout_from_dict(v) for k, v in spec["layouts"].items()}
    h = TransposeHarness(layouts, spec["tile"], spec["block_rows"]).build()
    x = torch.randn(spec["n"], spec["n"], device="cuda")
    for name in layouts:
        h.run(name, x)
    torch.cuda.synchronize()


if __name__ == "__main__":
    _main(sys.argv)
