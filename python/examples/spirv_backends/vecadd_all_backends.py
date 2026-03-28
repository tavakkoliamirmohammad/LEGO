#!/usr/bin/env python3
"""
Real vecadd (C = A + B) across all GPU backends.

Demonstrates:
  - KernelBuilder with multiple layout-aware buffers
  - lego.compile() unified API targeting: cuda, vulkan, webgpu, metal, webgl
  - Each buffer can have a different LEGO layout

Usage:
    python vecadd_all_backends.py
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import lego
from lego.core import Row, Col, OrderBy
from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer


def main():
    N = 256

    # ---- Define buffers with different LEGO layouts ----
    # A: column-major (non-trivial index arithmetic)
    # B: row-major
    # C: row-major (output)
    a = LayoutBuffer(Col(16, 16), shape=(16, 16), dtype="f32")
    b = LayoutBuffer(Row(N),      shape=(N,),      dtype="f32")
    c = LayoutBuffer(Row(N),      shape=(N,),      dtype="f32")

    # ---- Define the kernel body ----
    def vecadd_body(ctx):
        va = ctx.load(0)            # load A[gid] via Col layout
        vb = ctx.load(1)            # load B[gid] via Row layout
        ctx.store(ctx.add(va, vb), 2)  # store C[gid] = A + B

    builder = KernelBuilder(
        buffers=[a, b, c],
        kernel_body=vecadd_body,
        name="vecadd",
        workgroup_size=64,
    )

    # ---- Show generated MLIR ----
    ctx, module = builder.build_module()
    print("=" * 60)
    print("Generated MLIR (LEGO + GPU dialect)")
    print("=" * 60)
    mlir = str(module)
    for line in mlir.splitlines()[:30]:
        print(f"  {line}")
    print(f"  ... ({len(mlir.splitlines())} total lines)")
    print()

    # ---- Compile to all targets via unified API ----
    output_dir = os.path.join(_script_dir, "output")
    targets = ["cuda", "vulkan", "webgpu", "metal", "webgl"]

    for target in targets:
        print(f"--- {target} ---")
        try:
            result = builder.compile(
                target=target,
                output_dir=output_dir,
                name=f"vecadd_{target}",
            )
            print(f"  Output: {result.kernel_path}")
            if result.kernel_source and not result.kernel_source.startswith("<"):
                lines = result.kernel_source.splitlines()
                for line in lines[:10]:
                    print(f"    {line}")
                if len(lines) > 10:
                    print(f"    ... ({len(lines) - 10} more lines)")
            else:
                print(f"  {result.kernel_source}")
        except Exception as e:
            print(f"  FAILED: {e}")
        print()

    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
