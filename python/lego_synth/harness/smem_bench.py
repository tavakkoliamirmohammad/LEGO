"""Shared-memory-bound microbenchmark: time is proportional to predicted wavefronts.

Each thread (tx, ty) of a (tile, block_rows) block repeatedly loads the tile with
the column pattern (lane -> row), rotating the column each repetition so the
compiler cannot hoist the loads. With no global traffic in the loop, runtime is
dominated by shared-memory wavefronts.
"""
import torch  # noqa: F401  (hardware harness; torch is a hard requirement here)

from ..eval_np import bank_conflicts
from ..machine import ampere
from .cuda_emit import index_expr
import numpy as np


def predicted_wavefronts_per_instr(layout, machine=ampere()) -> int:
    """Wavefronts of one column-pattern load instruction (lane t reads (t, c))."""
    lanes = np.arange(machine.warp)
    addrs = layout.flat(lanes, np.zeros_like(lanes)) * 4
    return 1 + bank_conflicts(addrs, elem_bytes=4, machine=machine)


def kernel_source(name: str, layout, tile: int, block_rows: int) -> str:
    idx = index_expr(layout)
    return f"""
__global__ void smem_bench_{name}(const float* __restrict__ in, float* __restrict__ out, int reps) {{
  __shared__ float tile[{layout.footprint}];
  const int tx = threadIdx.x, ty = threadIdx.y;
  #pragma unroll
  for (int r = ty; r < {tile}; r += {block_rows}) {{
    const unsigned i = r, j = tx;
    tile[{idx}] = in[r * {tile} + tx];
  }}
  __syncthreads();
  float acc = 0.f;
  for (int it = 0; it < reps; ++it) {{
    #pragma unroll
    for (int r = ty; r < {tile}; r += {block_rows}) {{
      const unsigned i = tx, j = (r + it) & {tile - 1};
      acc += tile[{idx}];
    }}
  }}
  out[(blockIdx.x * {block_rows} + ty) * {tile} + tx] = acc;
}}

torch::Tensor bench_{name}(torch::Tensor x, int reps, int blocks) {{
  auto out = torch::empty({{blocks * {block_rows} * {tile}}}, x.options());
  dim3 grid(blocks), block({tile}, {block_rows});
  smem_bench_{name}<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(), reps);
  return out;
}}
"""


class SmemBench:
    def __init__(self, layouts, tile: int = 32, block_rows: int = 8):
        assert tile & (tile - 1) == 0, "tile must be a power of two (column rotation uses a mask)"
        self.layouts, self.tile, self.block_rows = dict(layouts), tile, block_rows
        self.module = None

    def build(self):
        from torch.utils.cpp_extension import load_inline
        cuda_src = "\n".join(kernel_source(n, L, self.tile, self.block_rows) for n, L in self.layouts.items())
        decls = "\n".join(f"torch::Tensor bench_{n}(torch::Tensor x, int reps, int blocks);" for n in self.layouts)
        key = abs(hash((cuda_src, self.tile, self.block_rows))) % (1 << 32)
        self.module = load_inline(name=f"lego_smem_bench_{key:08x}", cpp_sources=decls, cuda_sources=cuda_src,
                                  functions=[f"bench_{n}" for n in self.layouts], verbose=False)
        return self

    def time(self, name: str, reps: int = 1024, blocks: int = None, iters: int = 20, warmup: int = 3) -> float:
        if blocks is None:   # 8 blocks per SM on whatever GPU this is
            blocks = 8 * torch.cuda.get_device_properties(0).multi_processor_count
        fn = getattr(self.module, f"bench_{name}")
        x = torch.randn(self.tile * self.tile, device="cuda")
        for _ in range(warmup):
            fn(x, reps, blocks)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(x, reps, blocks); e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        return times[len(times) // 2]
