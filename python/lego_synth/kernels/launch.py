"""Launch CASTLE-compiled kernels on device tensors through CuPy (driver JIT of the PTX).

After `lego-to-nvvm`, every 1-D memref argument becomes a five-field descriptor
(allocated ptr, aligned ptr, offset, size, stride), so `launch` packs each tensor
as (ptr, ptr, 0, numel, 1). Grid and block come from the KernelBuilder.
"""
import os
import re
import numpy as np
import cupy
import torch


def extract_ptx(mlir_text: str) -> str:
    """PTX from the `assembly = "..."` string of a gpu.binary object (MLIR escapes as \XX hex)."""
    m = re.search(r'assembly = "((?:[^"\\]|\\.)*)"', mlir_text)
    if not m:
        raise ValueError("no assembly attribute in compiled module")
    return re.sub(r"\\([0-9A-Fa-f]{2})", lambda mm: chr(int(mm.group(1), 16)), m.group(1))


class CastleKernel:
    def __init__(self, ptx_path: str, entry: str, grid, block):
        self.ptx_path, self.entry, self.grid, self.block = ptx_path, entry, tuple(grid), tuple(block)
        self._fn = cupy.RawModule(path=ptx_path).get_function(entry)

    @classmethod
    def from_builder(cls, builder, chip: str = "sm_86", features: str = "+ptx78") -> "CastleKernel":
        res = builder.compile(target="cuda", chip=chip, features=features, format="assembly")
        ptx = extract_ptx(open(res.kernel_path).read())
        ptx_path = os.path.splitext(res.kernel_path)[0] + ".ptx"
        with open(ptx_path, "w") as f:
            f.write(ptx)
        entry = re.search(r"\.visible \.entry (\w+)\(", ptx).group(1)
        grid = tuple(builder._grid) + (1,) * (3 - len(builder._grid))
        block = tuple(builder._block) + (1,) * (3 - len(builder._block))
        return cls(ptx_path, entry, grid, block)

    @staticmethod
    def _descriptor(t: torch.Tensor):
        assert t.is_cuda and t.is_contiguous()
        p = np.uint64(t.data_ptr())
        return [p, p, np.int64(0), np.int64(t.numel()), np.int64(1)]

    def launch(self, *tensors: torch.Tensor):
        args = [a for t in tensors for a in self._descriptor(t)]
        self._fn(self.grid, self.block, tuple(args))

    def time(self, *tensors: torch.Tensor, iters: int = 100, warmup: int = 10) -> float:
        for _ in range(warmup):
            self.launch(*tensors)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); self.launch(*tensors); e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        return times[len(times) // 2]
