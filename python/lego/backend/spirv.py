"""
LEGO SPIR-V compilation backend.

Uses the shared gpu_builder for IR generation, then runs the
lego-to-spirv MLIR pipeline and optionally fans out via naga.

No GPU hardware required — this is a cross-compiler.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from lego.mlir.ir import Context, Location, Module
from lego.mlir.passmanager import PassManager

from lego.backend.dialects.lego_dialect import register as register_lego

# Re-export shared types so existing imports keep working
from lego.backend.gpu_builder import (  # noqa: F401
    CompileResult,
    LayoutBuffer,
    KernelBuilder,
    KernelContext,
    make_permutation_kernel,
)


# ============================================================================
# GPUIRBuilder — backward-compatible single-layout permutation builder
# ============================================================================

class GPUIRBuilder:
    """Builds a single-layout permutation kernel: dst[layout(gid)] = src[gid].

    For multi-buffer kernels with computation, use KernelBuilder instead.
    """

    def __init__(self, layout, shape, dtype_str="f32", workgroup_size=64):
        self._builder = make_permutation_kernel(layout, shape, dtype_str, workgroup_size)

    def build_module(self):
        return self._builder.build_module()


# ============================================================================
# SPIR-V pipeline
# ============================================================================

_TARGET_POST = {
    "vulkan": None,
    "webgpu": "wgsl",
    "metal":  "metal",
    "webgl":  "glsl",
}


def _extract_spirv_binary(module_str: str) -> Optional[List[int]]:
    """Extract SPIR-V binary from the 'lego.spirv_binary' string attribute."""
    import re
    match = re.search(r'lego\.spirv_binary\s*=\s*"([^"]*)"', module_str)
    if not match:
        return None
    values_str = match.group(1)
    return [int(v.strip()) for v in values_str.split(",") if v.strip()]


def _get_constant_value(value):
    """Extract a Python number from an MLIR Value defined by arith.constant."""
    from lego.mlir import ir

    if isinstance(value.owner, ir.Block):
        return None
    op = value.owner
    if op.name != "arith.constant":
        return None

    attr = op.attributes["value"]
    ty_str = str(value.type)
    if ty_str == "index" or ty_str.startswith("i"):
        return ("const_int", int(ir.IntegerAttr(attr)))
    elif ty_str.startswith("f"):
        return ("const_float", float(ir.FloatAttr(attr)))
    return None


def _extract_launch_metadata(module):
    """Extract dispatch metadata from a lowered SPIR-V module.

    Walks the MLIR operation tree to find gpu.launch_func and reads its
    operands to determine grid/thread dimensions and kernel argument bindings.

    Returns dict with keys: grid, threads, bindings, entry_point.
    """
    # Find gpu.launch_func by walking the module
    launch_op = None

    def _find_launch(op):
        nonlocal launch_op
        if launch_op is not None:
            return
        if op.name == "gpu.launch_func":
            launch_op = op
            return
        for region in op.regions:
            for block in region.blocks:
                for child in block:
                    _find_launch(child)
                    if launch_op is not None:
                        return

    _find_launch(module.operation)

    if launch_op is None:
        return {"grid": [1, 1, 1], "threads": [1, 1, 1],
                "bindings": [], "entry_point": ""}

    operands = list(launch_op.operands)

    # Determine operand layout from operandSegmentSizes attribute:
    #   [asyncDeps, grid_x, grid_y, grid_z,
    #    block_x, block_y, block_z, dynamicSharedMemSize, kernelOperands]
    seg_attr = launch_op.attributes.get("operandSegmentSizes")
    if seg_attr is not None:
        segs = [int(seg_attr[i]) for i in range(len(seg_attr))]
        async_count = segs[0]
        kernel_args_start = async_count + 6 + segs[7]
    else:
        async_count = 0
        kernel_args_start = 6

    # Grid dimensions
    grid = [1, 1, 1]
    for i in range(3):
        cv = _get_constant_value(operands[async_count + i])
        if cv is not None:
            grid[i] = cv[1]

    # Thread (block) dimensions
    threads = [1, 1, 1]
    for i in range(3):
        cv = _get_constant_value(operands[async_count + 3 + i])
        if cv is not None:
            threads[i] = cv[1]

    # Kernel argument bindings
    bindings = []
    buf_idx = 0
    for operand in operands[kernel_args_start:]:
        ty_str = str(operand.type)
        if "memref" in ty_str:
            bindings.append(("data", buf_idx))
            buf_idx += 1
        else:
            cv = _get_constant_value(operand)
            if cv is not None:
                bindings.append(cv)
            else:
                bindings.append(("const_int", 0))

    # Entry point name from kernel attribute
    # gpu.launch_func has @module::@func_name — we want the func_name part
    entry_point = ""
    kernel_attr = launch_op.attributes.get("kernel")
    if kernel_attr is not None:
        raw = str(kernel_attr)
        # Extract the last @name from patterns like @module::@entry
        parts = raw.split("@")
        entry_point = parts[-1].strip('"') if parts else raw.strip("@").strip('"')

    return {
        "grid": grid,
        "threads": threads,
        "bindings": bindings,
        "entry_point": entry_point,
    }


def compile_to_spirv(layout_or_builder, shape=None, dtype="f32", workgroup_size=64):
    """Compile to SPIR-V binary. No GPU hardware required.

    Args:
        layout_or_builder: A LEGO layout, KernelBuilder, or GPUIRBuilder.
        shape: Required if layout_or_builder is a layout.

    Returns:
        (spv_words, mlir_output, metadata) where:
          - spv_words: list of uint32 SPIR-V words
          - mlir_output: lowered MLIR text (for debugging)
          - metadata: dict with grid, threads, bindings, entry_point
    """
    if isinstance(layout_or_builder, KernelBuilder):
        builder = layout_or_builder
    elif isinstance(layout_or_builder, GPUIRBuilder):
        builder = layout_or_builder._builder
    else:
        builder = make_permutation_kernel(layout_or_builder, shape, dtype, workgroup_size)

    ctx, module = builder.build_module()

    with ctx:
        pm = PassManager.parse("builtin.module(lego-to-spirv)")
        try:
            pm.run(module.operation)
        except Exception as e:
            raise RuntimeError(f"lego-to-spirv pipeline failed:\n{e}") from e

    # Extract dispatch metadata from the live module before serializing to text
    metadata = _extract_launch_metadata(module)

    # Add workgroup buffer info from the builder (for Metal threadgroup alloc).
    # Workgroup args are lowered to SPIR-V globals by LowerWorkgroupToSPIRVPass
    # and no longer appear in the MLIR, so we get sizes from the builder.
    workgroups = []
    if hasattr(builder, '_shared_bufs'):
        for sbuf in builder._shared_bufs:
            workgroups.append(sbuf.numel)
    metadata["workgroups"] = workgroups

    # Extract binary: direct attribute access, fallback to regex
    spv_words = None
    try:
        attr = module.operation.attributes["lego.spirv_binary"]
        spv_words = [int(v) for v in str(attr).strip('"').split(",") if v.strip()]
    except (KeyError, ValueError):
        spv_words = _extract_spirv_binary(str(module))

    mlir_output = str(module)

    if not spv_words:
        raise RuntimeError(
            f"SPIR-V pipeline produced no lego.spirv_binary attribute.\n"
            f"Output IR:\n{mlir_output[:2000]}"
        )

    return spv_words, mlir_output, metadata


def compile_to_target(
    layout_or_builder,
    shape=None,
    target: str = "webgpu",
    dtype: str = "f32",
    output_dir: Optional[str] = None,
    name: str = "kernel",
    workgroup_size: int = 64,
) -> CompileResult:
    """Compile to the specified GPU target. No GPU hardware required."""
    if target not in _TARGET_POST:
        raise ValueError(
            f"Unknown target '{target}'. Supported: {list(_TARGET_POST.keys())}"
        )

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="lego_spirv_")
    os.makedirs(output_dir, exist_ok=True)

    spv_words, _, metadata = compile_to_spirv(layout_or_builder, shape, dtype, workgroup_size)

    from lego.backend.naga import spv_bytes_to_file
    spv_path = os.path.join(output_dir, f"{name}.spv")
    spv_bytes_to_file(spv_words, spv_path)

    post_fmt = _TARGET_POST[target]
    if post_fmt is None:
        size = os.path.getsize(spv_path)
        return CompileResult(
            target=target, kernel_path=spv_path,
            kernel_source=f"<binary SPIR-V: {size} bytes>", spv_path=spv_path,
            metadata=metadata,
        )

    try:
        from lego.backend import naga
    except ImportError:
        raise RuntimeError(f"naga not available. Cannot convert to {post_fmt}.")

    ext = {"wgsl": ".wgsl", "metal": ".metal", "glsl": ".comp"}[post_fmt]
    out_path = os.path.join(output_dir, f"{name}{ext}")

    try:
        if post_fmt == "wgsl":
            actual_path = naga.spv_to_wgsl(spv_path, out_path)
        elif post_fmt == "metal":
            actual_path = naga.spv_to_metal(spv_path, out_path)
        elif post_fmt == "glsl":
            actual_path = naga.spv_to_glsl(spv_path, out_path)
    except FileNotFoundError:
        raise RuntimeError(
            "naga binary not found. Install: cargo install naga-cli"
        )

    kernel_source = Path(actual_path).read_text()
    return CompileResult(
        target=target, kernel_path=actual_path,
        kernel_source=kernel_source, spv_path=spv_path,
        metadata=metadata,
    )


def compile_all(
    layout_or_builder,
    shape=None,
    targets: Optional[List[str]] = None,
    dtype: str = "f32",
    output_dir: Optional[str] = None,
    name: str = "kernel",
    workgroup_size: int = 64,
) -> Dict[str, CompileResult]:
    """Compile to multiple GPU targets. Builds SPIR-V once, fans out via naga."""
    if targets is None:
        targets = ["vulkan", "webgpu", "metal", "webgl"]

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="lego_multi_")
    os.makedirs(output_dir, exist_ok=True)

    spv_words, _, metadata = compile_to_spirv(layout_or_builder, shape, dtype, workgroup_size)

    from lego.backend.naga import spv_bytes_to_file
    spv_path = os.path.join(output_dir, f"{name}.spv")
    spv_bytes_to_file(spv_words, spv_path)

    results = {}
    for target in targets:
        post_fmt = _TARGET_POST.get(target)
        if post_fmt is None:
            size = os.path.getsize(spv_path)
            results[target] = CompileResult(
                target=target, kernel_path=spv_path,
                kernel_source=f"<binary SPIR-V: {size} bytes>", spv_path=spv_path,
                metadata=metadata,
            )
            continue

        from lego.backend import naga
        ext = {"wgsl": ".wgsl", "metal": ".metal", "glsl": ".comp"}[post_fmt]
        out_path = os.path.join(output_dir, f"{name}{ext}")

        if post_fmt == "wgsl":
            actual_path = naga.spv_to_wgsl(spv_path, out_path)
        elif post_fmt == "metal":
            actual_path = naga.spv_to_metal(spv_path, out_path)
        elif post_fmt == "glsl":
            actual_path = naga.spv_to_glsl(spv_path, out_path)

        kernel_source = Path(actual_path).read_text()
        results[target] = CompileResult(
            target=target, kernel_path=actual_path,
            kernel_source=kernel_source, spv_path=spv_path,
            metadata=metadata,
        )

    return results
