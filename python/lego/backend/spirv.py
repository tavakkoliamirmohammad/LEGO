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

from mlir.ir import Context, Location, Module
from mlir.passmanager import PassManager

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


def compile_to_spirv(layout_or_builder, shape=None, dtype="f32", workgroup_size=64):
    """Compile to SPIR-V binary. No GPU hardware required.

    Args:
        layout_or_builder: A LEGO layout, KernelBuilder, or GPUIRBuilder.
        shape: Required if layout_or_builder is a layout.

    Returns:
        (spv_words, mlir_output) where spv_words is a list of uint32.
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

    return spv_words, mlir_output


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

    spv_words, _ = compile_to_spirv(layout_or_builder, shape, dtype, workgroup_size)

    from lego.backend.naga import spv_bytes_to_file
    spv_path = os.path.join(output_dir, f"{name}.spv")
    spv_bytes_to_file(spv_words, spv_path)

    post_fmt = _TARGET_POST[target]
    if post_fmt is None:
        size = os.path.getsize(spv_path)
        return CompileResult(
            target=target, kernel_path=spv_path,
            kernel_source=f"<binary SPIR-V: {size} bytes>", spv_path=spv_path,
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

    spv_words, _ = compile_to_spirv(layout_or_builder, shape, dtype, workgroup_size)

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
        )

    return results
