"""ldmatrix.x4 microbenchmark on a 16-bit shared-memory tile under a given layout.

Per variant two entries are built: `check` performs one ldmatrix.x4 per 16x16 block
with a single warp and returns the four fragment registers of every lane, so the
lane mapping can be verified against the PTX ISA; `bench` repeats the loads in a
loop with several warps per block so runtime is dominated by shared-memory wavefronts.
"""
import numpy as np
import torch

from ..eval_np import bank_conflicts
from ..machine import ampere
from ..patterns import ldmatrix_x4
from .cuda_emit import index_expr


def predicted_wavefronts_per_ldmatrix(layout, machine=ampere()) -> int:
    """Total wavefronts of one ldmatrix.x4 on the tile's first 16x16 block: four 8-lane
    phases (one per 8x8 matrix) plus the extra wavefronts the model predicts."""
    pat = ldmatrix_x4(16, 16)
    i, j = pat.coords(0)
    addrs = layout.flat(i, j) * 2
    return 4 + bank_conflicts(addrs, elem_bytes=16, machine=machine)


def kernel_source(name: str, layout, rows: int, cols: int, warps: int) -> str:
    idx = index_expr(layout)
    nb_r, nb_c = rows // 16, cols // 16
    return f"""
__device__ __forceinline__ void ldm_x4_{name}(const unsigned short* tile, unsigned lane, unsigned r0, unsigned c0,
                                              unsigned& a, unsigned& b, unsigned& c, unsigned& d) {{
  const unsigned m = lane >> 3, r = lane & 7;
  const unsigned i = r0 + r + 8u * (m & 1u), j = c0 + 8u * (m >> 1);
  const unsigned addr = (unsigned)__cvta_generic_to_shared(tile + {idx});
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%0,%1,%2,%3}}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}}

__device__ __forceinline__ void fill_{name}(unsigned short* tile) {{
  for (unsigned e = threadIdx.x; e < {rows * cols}; e += blockDim.x) {{
    const unsigned i = e / {cols}, j = e % {cols};
    tile[{idx}] = (unsigned short)(i * {cols} + j);
  }}
  __syncthreads();
}}

__global__ void ldm_check_{name}(unsigned* out) {{
  __shared__ __align__(16) unsigned short tile[{layout.footprint}];
  fill_{name}(tile);
  const unsigned lane = threadIdx.x;
  for (unsigned br = 0; br < {nb_r}; ++br)
    for (unsigned bc = 0; bc < {nb_c}; ++bc) {{
      unsigned a, b, c, d;
      ldm_x4_{name}(tile, lane, br * 16, bc * 16, a, b, c, d);
      unsigned* o = out + ((br * {nb_c} + bc) * 32 + lane) * 4;
      o[0] = a; o[1] = b; o[2] = c; o[3] = d;
    }}
}}

__global__ void ldm_bench_{name}(unsigned* out, int reps) {{
  __shared__ __align__(16) unsigned short tile[{layout.footprint}];
  fill_{name}(tile);
  const unsigned lane = threadIdx.x & 31;
  unsigned acc = 0;
  for (int it = 0; it < reps; ++it) {{
    #pragma unroll
    for (unsigned br = 0; br < {nb_r}; ++br)
      #pragma unroll
      for (unsigned bc = 0; bc < {nb_c}; ++bc) {{
        unsigned a, b, c, d;
        ldm_x4_{name}(tile, lane, br * 16, bc * 16, a, b, c, d);
        acc ^= a ^ b ^ c ^ d;
      }}
  }}
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}}

torch::Tensor check_{name}() {{
  auto out = torch::empty({{{nb_r * nb_c}, 32, 4}}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  ldm_check_{name}<<<1, 32>>>((unsigned*)out.data_ptr<int>());
  return out;
}}

torch::Tensor bench_{name}(int reps, int blocks) {{
  auto out = torch::empty({{blocks * {warps * 32}}}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  ldm_bench_{name}<<<blocks, {warps * 32}>>>((unsigned*)out.data_ptr<int>(), reps);
  return out;
}}
"""


def expected_fragments(rows: int, cols: int) -> np.ndarray:
    """Register contents ldmatrix.x4 must return when tile[i][j] == i*cols + j.

    For matrix m, lane t holds row t//4, columns 2*(t%4) and 2*(t%4)+1 packed low/high.
    """
    nb_r, nb_c = rows // 16, cols // 16
    out = np.zeros((nb_r * nb_c, 32, 4), dtype=np.uint32)
    for br in range(nb_r):
        for bc in range(nb_c):
            for m in range(4):
                mr, mc = br * 16 + 8 * (m & 1), bc * 16 + 8 * (m >> 1)
                for t in range(32):
                    i, j = mr + t // 4, mc + 2 * (t % 4)
                    lo, hi = i * cols + j, i * cols + j + 1
                    out[br * nb_c + bc, t, m] = (hi << 16) | lo
    return out


class LdmatrixBench:
    def __init__(self, layouts, rows: int, cols: int, warps: int = 4):
        assert rows % 16 == 0 and cols % 16 == 0
        self.layouts, self.rows, self.cols, self.warps = dict(layouts), rows, cols, warps
        self.module = None

    def build(self):
        from torch.utils.cpp_extension import load_inline
        cuda_src = "\n".join(kernel_source(n, L, self.rows, self.cols, self.warps) for n, L in self.layouts.items())
        decls = "\n".join(f"torch::Tensor check_{n}();\ntorch::Tensor bench_{n}(int reps, int blocks);" for n in self.layouts)
        key = abs(hash((cuda_src, self.rows, self.cols, self.warps))) % (1 << 32)
        fns = [f"check_{n}" for n in self.layouts] + [f"bench_{n}" for n in self.layouts]
        self.module = load_inline(name=f"lego_ldmatrix_{key:08x}", cpp_sources=decls, cuda_sources=cuda_src,
                                  functions=fns, verbose=False)
        return self

    def check(self, name: str) -> bool:
        got = getattr(self.module, f"check_{name}")().cpu().numpy().astype(np.uint32)
        return np.array_equal(got, expected_fragments(self.rows, self.cols))

    def time(self, name: str, reps: int = 1024, blocks: int = None, iters: int = 20, warmup: int = 3) -> float:
        if blocks is None:   # 4 blocks per SM on whatever GPU this is
            blocks = 4 * torch.cuda.get_device_properties(0).multi_processor_count
        fn = getattr(self.module, f"bench_{name}")
        for _ in range(warmup):
            fn(reps, blocks)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(reps, blocks); e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        return times[len(times) // 2]
