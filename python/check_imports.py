"""Verify that all LEGO Python imports work correctly."""
print("[check-lego-imports] Testing core imports...")
from lego import jit
from lego.core import OrderBy, Row, Col, GroupBy
print("[check-lego-imports] Testing backend imports...")
from lego.backend.compiler import LayoutCompiler
from lego.backend.dialects.lego_dialect import register
print("[check-lego-imports] Testing frontend imports...")
from lego.frontends._adapter import DSLAdapter
from lego.frontends.python_mlir import (
    TiledPermute, RowMajor, ColMajor, Custom, Transposed,
    ZCurve, Swizzle, BlockCyclic, Batched, LegoLayout, LegoArray,
)
from lego.frontends.triton_jit import TritonAdapter, jit as triton_jit
from lego.frontends.numba_jit import NumbaCUDAAdapter, jit as numba_jit
from lego.frontends.jax_jit import JAXAdapter, jit as jax_jit
from lego.frontends.cutile_jit import CutileAdapter, cutile_jit
from lego.frontends.rust_gen import RustAdapter, generate as rust_generate
from lego.frontends.cxx_gen import CXXAdapter, generate as cxx_generate
from lego.frontends.fortran_gen import FortranAdapter, generate as fortran_generate
from lego.frontends.cuda_c_gen import CUDACAdapter, generate as cuda_c_generate
from lego.frontends.julia_gen import JuliaAdapter, generate as julia_generate
from lego.frontends.js_gen import JSAdapter, generate as js_generate
from lego.frontends.glsl_gen import GLSLAdapter, generate as glsl_generate
from lego.core import le_constraint, divisibility_constraint
print("[check-lego-imports] Testing GPU builder + SPIR-V backend imports...")
from lego.backend.gpu_builder import KernelBuilder, KernelContext, LayoutBuffer, CompileResult, make_permutation_kernel
from lego.backend.spirv import GPUIRBuilder, compile_to_spirv, compile_to_target, compile_all
from lego.backend.naga import convert, spv_to_wgsl, spv_to_metal, spv_to_glsl, spv_bytes_to_file, validate
print("[check-lego-imports] All imports OK")
