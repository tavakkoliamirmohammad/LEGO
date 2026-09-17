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
    r"""PTX from the `assembly = "..."` string of a gpu.binary object (MLIR escapes as \XX hex)."""
    m = re.search(r'assembly = "((?:[^"\\]|\\.)*)"', mlir_text)
    if not m:
        raise ValueError("no assembly attribute in compiled module")
    return re.sub(r"\\([0-9A-Fa-f]{2})", lambda mm: chr(int(mm.group(1), 16)), m.group(1))


class CastleKernel:
    def __init__(self, ptx_path: str, entry: str, grid, block, shared_bytes: int = 0):
        self.ptx_path, self.entry, self.grid, self.block = ptx_path, entry, tuple(grid), tuple(block)
        self.shared_bytes = int(shared_bytes)
        self._fn = cupy.RawModule(path=ptx_path).get_function(entry)
        if self.shared_bytes > 48 * 1024:
            self._fn.max_dynamic_shared_size_bytes = self.shared_bytes

    @classmethod
    def from_builder(cls, builder, chip: str = None, features: str = None, max_regs: int = None) -> "CastleKernel":
        """Compile for the detected GPU unless a chip / PTX feature string is given.

        max_regs adds a `.maxnreg` directive to the entry: ptxas then schedules (and, if it must,
        spills) within that budget instead of hoisting every fragment load as early as registers allow."""
        if chip is None or features is None:
            from .target import detect_target
            t = detect_target()
            chip, features = chip or t.chip, features or t.features
        res = builder.compile(target="cuda", chip=chip, features=features, format="assembly")
        ptx = extract_ptx(open(res.kernel_path).read())
        if max_regs:
            ptx = re.sub(r"(\.entry\s+\w+\s*\([^)]*\)\s*)", lambda m: m.group(1) + f".maxnreg {int(max_regs)}\n", ptx, count=1)
        ptx_path = os.path.splitext(res.kernel_path)[0] + ".ptx"
        with open(ptx_path, "w") as f:
            f.write(ptx)
        entry = re.search(r"\.visible \.entry (\w+)\(", ptx).group(1)
        grid = tuple(builder._grid) + (1,) * (3 - len(builder._grid))
        block = tuple(builder._block) + (1,) * (3 - len(builder._block))
        return cls(ptx_path, entry, grid, block, shared_bytes=getattr(builder, "dynamic_shared_bytes", 0))

    @staticmethod
    def _descriptor(t: torch.Tensor):
        assert t.is_cuda and t.is_contiguous()
        p = np.uint64(t.data_ptr())
        return [p, p, np.int64(0), np.int64(t.numel()), np.int64(1)]

    def launch(self, *tensors: torch.Tensor):
        args = [a for t in tensors for a in self._descriptor(t)]
        self._fn(self.grid, self.block, tuple(args), shared_mem=self.shared_bytes)

    def time(self, *tensors: torch.Tensor, iters: int = 100, warmup: int = 10, batches: int = 3) -> float:
        """Median over batches of `iters` back-to-back launches (amortizes launch overhead)."""
        for _ in range(warmup):
            self.launch(*tensors)
        torch.cuda.synchronize()
        per = []
        for _ in range(batches):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                self.launch(*tensors)
            e.record(); torch.cuda.synchronize()
            per.append(s.elapsed_time(e) / iters)
        per.sort()
        return per[len(per) // 2]
