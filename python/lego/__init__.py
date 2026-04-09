"""
LEGO: Layout Expression Language for Code Generation

  Backend:   lego.backend (symbolic, codegen, compiler, dialects)
  Frontends: lego.frontends.triton_jit, .cutile_jit, .python_mlir
  PyTorch:   lego.torch (annotate, rearrange, LegoTensor, torch.compile)
"""
from .core import *
from .frontends.triton_jit import jit, get_kernel_source
from .frontends.cutile_jit import cutile_jit, get_cutile_kernel_source
from .frontends.python_mlir import (
    LegoLayout, RowMajor, ColMajor, TiledPermute, Custom,
    Transposed, ZCurve, Swizzle, BlockCyclic,
    Batched, BatchedLayout, LegoArray,
    row, col, reg_p, order_by, tile_by, group_by, gen_p,
)
from .frontends import rust_gen, fortran_gen, cxx_gen
from .frontends import julia_gen, cuda_c_gen, js_gen, glsl_gen

# Unified compile API — dispatches to CPU JIT, GPU pipeline, or SPIR-V
from .backend.gpu_builder import (
    KernelBuilder, LayoutBuffer, CompileResult,
    GPUTarget, _GPU_TARGETS,
)
from .backend.spirv import (
    compile_to_target as _compile_gpu,
    compile_to_spirv as _compile_spirv,
    compile_all as compile_all,
)
from .backend.compiler import DType, get_compiler as _get_cpu_compiler

_SPIRV_TARGETS = {"vulkan", "webgpu", "metal", "webgl"}


def _all_targets():
    return {"cpu", "wasm"} | set(_GPU_TARGETS) | _SPIRV_TARGETS


def compile(layout_or_builder, shape=None, target="cpu", dtype="f32", **kwargs):
    """Compile a LEGO layout to the specified target.

    Args:
        layout_or_builder: LEGO layout object, KernelBuilder, or GPUIRBuilder.
        shape: Tensor shape (required for layout input).
        target: Any registered target — "cpu", "cuda", "rocm",
                "vulkan", "webgpu", "metal", "webgl", or custom.
        dtype: Element type.
        **kwargs: Forwarded to the backend (chip, format, output_dir, …).

    Returns:
        For CPU: LayoutCompiler (use .transform_numpy(arr) to execute).
        For GPU: CompileResult with kernel_path and kernel_source.
    """
    if target == "cpu":
        return _get_cpu_compiler(layout_or_builder, shape, dtype)
    if target in _GPU_TARGETS:
        if isinstance(layout_or_builder, KernelBuilder):
            return layout_or_builder.compile(target=target, **kwargs)
        from lego.backend.gpu_builder import make_permutation_kernel
        builder = make_permutation_kernel(layout_or_builder, shape, dtype,
                                          kwargs.pop("workgroup_size", 64))
        return builder.compile(target=target, **kwargs)
    if target in _SPIRV_TARGETS:
        return _compile_gpu(layout_or_builder, shape=shape, target=target,
                            dtype=dtype, **kwargs)
    raise ValueError(
        f"Unknown target '{target}'. "
        f"Supported: {', '.join(sorted(_all_targets()))}"
    )


# PyTorch integration — only when torch is already loaded.
import sys as _sys3
if 'torch' in _sys3.modules:
    from .torch import annotate, rearrange, LegoTensor  # noqa: F401
del _sys3
