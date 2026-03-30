"""Comprehensive tests for the SPIR-V multi-target backend.

Tests are organized in four tiers:
  1. GPUIRBuilder — verify MLIR generation with gpu.launch + LEGO ops
  2. SPIR-V pipeline — verify lego-to-spirv produces valid SPIR-V binary
  3. Naga conversion — verify .spv → .wgsl / .metal / .glsl
  4. Multi-target compile — verify compile_to_target() and compile_all()

No GPU hardware required — all tests are pure cross-compilation.
"""

import os
import shutil
import struct
import tempfile
import pytest

from lego.core import Row, Col, GroupBy, OrderBy
from lego.backend.spirv import KernelBuilder, LayoutBuffer


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _has_lego_to_spirv_pipeline():
    """Check if the lego-to-spirv pipeline is available (requires SPIR-V libs)."""
    try:
        from lego.mlir.ir import Context, Location, Module
        from lego.mlir.passmanager import PassManager
        from lego.backend.dialects.lego_dialect import register
        ctx = Context()
        register(ctx)
        with ctx, Location.unknown():
            pm = PassManager.parse("builtin.module(lego-to-spirv)")
        return True
    except Exception:
        return False


def _has_lego_to_llvmspirv_pipeline():
    """Check if the lego-to-llvmspirv pipeline is available."""
    try:
        from lego.mlir.ir import Context, Location, Module
        from lego.mlir.passmanager import PassManager
        from lego.backend.dialects.lego_dialect import register
        ctx = Context()
        register(ctx)
        ctx.load_all_available_dialects()
        with ctx, Location.unknown():
            pm = PassManager.parse("builtin.module(lego-to-llvmspirv{chip=generic format=assembly})")
        return True
    except Exception:
        return False


def _has_naga():
    """Check if the naga binary is available."""
    try:
        from lego.backend.naga import _find_naga_binary
        _find_naga_binary()
        return True
    except FileNotFoundError:
        return False


requires_spirv = pytest.mark.skipif(
    not _has_lego_to_spirv_pipeline(),
    reason="lego-to-spirv pipeline not available (SPIR-V libs not built)",
)

requires_llvmspirv = pytest.mark.skipif(
    not _has_lego_to_llvmspirv_pipeline(),
    reason="lego-to-llvmspirv pipeline not available",
)

requires_naga = pytest.mark.skipif(
    not _has_naga(),
    reason="naga binary not found (install with: cargo install naga-cli)",
)

requires_ptxas = pytest.mark.skipif(
    shutil.which("ptxas") is None,
    reason="ptxas not found in PATH (load a CUDA toolkit module)",
)


# ══════════════════════════════════════════════════════════════════════════
# Tier 1: GPUIRBuilder — MLIR generation
# ══════════════════════════════════════════════════════════════════════════

class TestGPUIRBuilder:
    """Test that GPUIRBuilder produces valid MLIR with gpu.launch + LEGO ops."""

    @requires_spirv
    def test_row_layout_builds(self):
        from lego.backend.spirv import GPUIRBuilder
        builder = GPUIRBuilder(Row(256), shape=(256,), workgroup_size=64)
        ctx, module = builder.build_module()
        mlir = str(module)
        assert "gpu.launch" in mlir
        assert "lego." in mlir  # LEGO ops present (row, apply, group_by, etc.)
        assert "func.func" in mlir
        assert "gpu.launch" in mlir

    @requires_spirv
    def test_col_layout_builds(self):
        from lego.backend.spirv import GPUIRBuilder
        builder = GPUIRBuilder(Col(16, 16), shape=(16, 16), workgroup_size=64)
        ctx, module = builder.build_module()
        mlir = str(module)
        assert "gpu.launch" in mlir
        assert "lego.col" in mlir

    @requires_spirv
    def test_tiled_layout_builds(self):
        from lego.backend.spirv import GPUIRBuilder
        layout = OrderBy(Row(64, 64)).TileBy([16, 16])
        builder = GPUIRBuilder(layout, shape=(64, 64), workgroup_size=256)
        ctx, module = builder.build_module()
        mlir = str(module)
        assert "gpu.launch" in mlir

    @requires_spirv
    def test_workgroup_size_respected(self):
        from lego.backend.spirv import GPUIRBuilder
        builder = GPUIRBuilder(Row(512), shape=(512,), workgroup_size=128)
        ctx, module = builder.build_module()
        mlir = str(module)
        # Grid = ceil(512 / 128) = 4, block = 128
        assert "128" in mlir

    @requires_spirv
    def test_bounds_check_present(self):
        from lego.backend.spirv import GPUIRBuilder
        builder = GPUIRBuilder(Row(100), shape=(100,), workgroup_size=64)
        ctx, module = builder.build_module()
        mlir = str(module)
        # Should have a bounds check (arith.cmpi ult)
        assert "arith.cmpi" in mlir or "cmpi" in mlir


# ══════════════════════════════════════════════════════════════════════════
# Tier 2: SPIR-V pipeline — compile_to_spirv
# ══════════════════════════════════════════════════════════════════════════

class TestSPIRVPipeline:
    """Test that the lego-to-spirv pipeline produces valid SPIR-V binary."""

    @requires_spirv
    def test_row_layout_compiles(self):
        from lego.backend.spirv import compile_to_spirv
        words, mlir = compile_to_spirv(Row(256), shape=(256,))
        assert len(words) > 10
        # SPIR-V magic number
        assert words[0] == 0x07230203

    @requires_spirv
    def test_col_layout_compiles(self):
        from lego.backend.spirv import compile_to_spirv
        words, _ = compile_to_spirv(Col(16, 16), shape=(16, 16))
        assert words[0] == 0x07230203

    @requires_spirv
    def test_2d_row_layout_compiles(self):
        from lego.backend.spirv import compile_to_spirv
        layout = Row(16, 16)
        words, _ = compile_to_spirv(layout, shape=(16, 16))
        assert words[0] == 0x07230203

    @requires_spirv
    def test_2d_col_layout_compiles(self):
        from lego.backend.spirv import compile_to_spirv
        layout = Col(8, 8)
        words, _ = compile_to_spirv(layout, shape=(8, 8))
        assert words[0] == 0x07230203

    @requires_spirv
    def test_tileby_layout_compiles(self):
        """TileBy: tile 8x8 into 4x4 blocks → two-level [4,4],[2,2]."""
        from lego.backend.spirv import compile_to_spirv
        layout = OrderBy(Row(8, 8)).TileBy([4, 4], [2, 2])
        words, _ = compile_to_spirv(layout, shape=(8, 8))
        assert words[0] == 0x07230203
        assert len(words) > 100  # TileBy generates more index math

    @requires_spirv
    def test_spirv_binary_attribute_in_output(self):
        from lego.backend.spirv import compile_to_spirv
        _, mlir = compile_to_spirv(Row(64), shape=(64,))
        assert "lego.spirv_binary" in mlir

    @requires_spirv
    def test_different_dtypes(self):
        from lego.backend.spirv import compile_to_spirv
        # f32 and i32 should work; f64 needs Float64 capability (future work)
        for dtype in ["f32", "i32"]:
            words, _ = compile_to_spirv(Row(64), shape=(64,), dtype=dtype)
            assert words[0] == 0x07230203, f"Failed for dtype={dtype}"


# ══════════════════════════════════════════════════════════════════════════
# Tier 3: Naga conversion — .spv → .wgsl / .metal / .glsl
# ══════════════════════════════════════════════════════════════════════════

class TestNagaConversion:
    """Test naga binary converts SPIR-V to all target formats."""

    @requires_spirv
    @requires_naga
    def test_spv_to_wgsl(self):
        from lego.backend.spirv import compile_to_spirv
        from lego.backend.naga import spv_bytes_to_file, spv_to_wgsl
        words, _ = compile_to_spirv(Row(256), shape=(256,))
        with tempfile.TemporaryDirectory() as d:
            spv_path = spv_bytes_to_file(words, os.path.join(d, "k.spv"))
            wgsl_path = spv_to_wgsl(spv_path)
            wgsl = open(wgsl_path).read()
            assert "@compute" in wgsl
            assert "@workgroup_size" in wgsl

    @requires_spirv
    @requires_naga
    def test_spv_to_metal(self):
        from lego.backend.spirv import compile_to_spirv
        from lego.backend.naga import spv_bytes_to_file, spv_to_metal
        words, _ = compile_to_spirv(Row(256), shape=(256,))
        with tempfile.TemporaryDirectory() as d:
            spv_path = spv_bytes_to_file(words, os.path.join(d, "k.spv"))
            metal_path = spv_to_metal(spv_path)
            msl = open(metal_path).read()
            assert "kernel void" in msl or "metal" in msl

    @requires_spirv
    @requires_naga
    def test_spv_validates(self):
        from lego.backend.spirv import compile_to_spirv
        from lego.backend.naga import spv_bytes_to_file, validate
        words, _ = compile_to_spirv(Row(256), shape=(256,))
        with tempfile.TemporaryDirectory() as d:
            spv_path = spv_bytes_to_file(words, os.path.join(d, "k.spv"))
            assert validate(spv_path)

    @requires_naga
    def test_naga_import(self):
        from lego.backend.naga import _find_naga_binary
        path = _find_naga_binary()
        assert os.path.isfile(path)


# ══════════════════════════════════════════════════════════════════════════
# Tier 4: Multi-target compile API
# ══════════════════════════════════════════════════════════════════════════

class TestMultiTargetCompile:
    """Test the high-level compile_to_target() and compile_all() API."""

    @requires_spirv
    @requires_naga
    def test_compile_to_vulkan(self):
        from lego.backend.spirv import compile_to_target
        result = compile_to_target(Row(256), shape=(256,), target="vulkan")
        assert result.target == "vulkan"
        assert result.spv_path.endswith(".spv")
        assert os.path.exists(result.spv_path)
        assert os.path.getsize(result.spv_path) > 0

    @requires_spirv
    @requires_naga
    def test_compile_to_webgpu(self):
        from lego.backend.spirv import compile_to_target
        result = compile_to_target(Row(256), shape=(256,), target="webgpu")
        assert result.target == "webgpu"
        assert result.kernel_path.endswith(".wgsl")
        assert "@compute" in result.kernel_source

    @requires_spirv
    @requires_naga
    def test_compile_to_webgl(self):
        from lego.backend.spirv import compile_to_target
        result = compile_to_target(Row(256), shape=(256,), target="webgl")
        assert result.target == "webgl"
        assert result.kernel_path.endswith(".comp")
        assert "layout(local_size" in result.kernel_source
        assert "void main()" in result.kernel_source

    @requires_spirv
    @requires_naga
    def test_compile_to_metal(self):
        from lego.backend.spirv import compile_to_target
        result = compile_to_target(Row(256), shape=(256,), target="metal")
        assert result.target == "metal"
        assert result.kernel_path.endswith(".metal")
        assert "kernel void" in result.kernel_source or "metal" in result.kernel_source

    @requires_spirv
    @requires_naga
    def test_compile_all_produces_all_targets(self):
        from lego.backend.spirv import compile_all
        results = compile_all(Row(128), shape=(128,))
        assert "vulkan" in results
        assert "webgpu" in results
        assert "metal" in results
        assert "webgl" in results
        for target, result in results.items():
            assert os.path.exists(result.kernel_path), f"{target} output missing"

    @requires_spirv
    @requires_naga
    def test_compile_all_reuses_spv(self):
        """All targets should share the same .spv file."""
        from lego.backend.spirv import compile_all
        results = compile_all(Row(128), shape=(128,), targets=["vulkan", "webgpu", "metal"])
        spv_paths = {r.spv_path for r in results.values()}
        assert len(spv_paths) == 1, "All targets should share one .spv"

    @requires_spirv
    @requires_naga
    def test_2d_col_layout_all_backends(self):
        from lego.backend.spirv import compile_all
        layout = Col(8, 8)
        results = compile_all(layout, shape=(8, 8), targets=["vulkan", "webgpu", "metal"])
        for target, result in results.items():
            assert os.path.exists(result.kernel_path), f"{target} missing"

    @requires_spirv
    @requires_naga
    def test_tileby_all_backends(self):
        """TileBy layout compiled to all GPU backends."""
        from lego.backend.spirv import compile_all
        layout = OrderBy(Row(8, 8)).TileBy([4, 4], [2, 2])
        results = compile_all(layout, shape=(8, 8), targets=["vulkan", "webgpu", "metal"])
        for target, result in results.items():
            assert os.path.exists(result.kernel_path), f"{target} missing"
        # WGSL should contain real index arithmetic
        assert ">>" in results["webgpu"].kernel_source or "&" in results["webgpu"].kernel_source

    @requires_spirv
    def test_invalid_target_raises(self):
        from lego.backend.spirv import compile_to_target
        with pytest.raises(ValueError, match="Unknown target"):
            compile_to_target(Row(64), shape=(64,), target="directx12")

    @requires_spirv
    def test_spirv_magic_number(self):
        """SPIR-V binary must start with the magic number 0x07230203."""
        from lego.backend.spirv import compile_to_spirv
        words, _ = compile_to_spirv(Row(128), shape=(128,))
        assert words[0] == 0x07230203

    @requires_spirv
    def test_workgroup_size_1(self):
        """Edge case: workgroup_size=1 should still produce valid SPIR-V."""
        from lego.backend.spirv import compile_to_spirv
        words, _ = compile_to_spirv(Row(16), shape=(16,), workgroup_size=1)
        assert words[0] == 0x07230203

    @requires_spirv
    def test_non_power_of_2_shape(self):
        """Non-power-of-2 shapes should compile correctly."""
        from lego.backend.spirv import compile_to_spirv
        words, _ = compile_to_spirv(Row(100), shape=(100,))
        assert words[0] == 0x07230203

    @requires_spirv
    @requires_naga
    def test_custom_output_dir(self):
        from lego.backend.spirv import compile_to_target
        with tempfile.TemporaryDirectory() as d:
            result = compile_to_target(
                Row(64), shape=(64,), target="webgpu",
                output_dir=d, name="my_kernel",
            )
            assert "my_kernel.wgsl" in result.kernel_path
            assert os.path.isfile(result.kernel_path)


# ══════════════════════════════════════════════════════════════════════════
# Tier 5: KernelBuilder — multi-buffer kernels with user-defined body
# ══════════════════════════════════════════════════════════════════════════

class TestKernelBuilder:
    """Test KernelBuilder with multi-buffer, user-defined kernel bodies."""

    @requires_spirv
    def test_vecadd_two_inputs_one_output(self):
        """Real vecadd: C[i] = A[i] + B[i] with different layouts."""
        from lego.backend.spirv import compile_to_spirv
        from lego.backend.gpu_builder import make_elementwise_kernel

        a = LayoutBuffer(Row(64), shape=(64,))
        b = LayoutBuffer(Col(8, 8), shape=(8, 8))
        c = LayoutBuffer(Row(64), shape=(64,))

        def vecadd(ctx, gid):
            va = ctx.load_flat(0, gid)
            vb = ctx.load_flat(1, gid)
            ctx.store_flat(ctx.addf(va, vb), 2, gid)

        builder = make_elementwise_kernel([a, b, c], vecadd, name="vecadd")
        words, _ = compile_to_spirv(builder)
        assert words[0] == 0x07230203

    @requires_spirv
    @requires_naga
    def test_vecadd_to_wgsl(self):
        """Vecadd compiled to WGSL."""
        from lego.backend.spirv import compile_to_target
        from lego.backend.gpu_builder import make_elementwise_kernel

        a = LayoutBuffer(Row(64), shape=(64,))
        b = LayoutBuffer(Row(64), shape=(64,))
        c = LayoutBuffer(Row(64), shape=(64,))

        def vecadd(ctx, gid):
            ctx.store_flat(ctx.addf(ctx.load_flat(0, gid), ctx.load_flat(1, gid)), 2, gid)

        builder = make_elementwise_kernel([a, b, c], vecadd, name="vecadd")
        result = compile_to_target(builder, target="webgpu")
        assert "@compute" in result.kernel_source

    @requires_spirv
    def test_scale_kernel(self):
        """Scale: B[i] = A[i] * A[i] (single input, single output)."""
        from lego.backend.spirv import compile_to_spirv
        from lego.backend.gpu_builder import make_elementwise_kernel

        a = LayoutBuffer(Col(4, 4), shape=(4, 4))
        b = LayoutBuffer(Row(4, 4), shape=(4, 4))

        def scale(ctx, gid):
            v = ctx.load_flat(0, gid)
            ctx.store_flat(ctx.mulf(v, v), 1, gid)

        builder = make_elementwise_kernel([a, b], scale, name="square")
        words, _ = compile_to_spirv(builder)
        assert words[0] == 0x07230203

    @requires_spirv
    def test_kernel_with_layout_indices(self):
        """Transpose-style: load with one layout, store with another."""
        from lego.backend.spirv import compile_to_spirv

        src = LayoutBuffer(Row(4, 4), shape=(4, 4))
        dst = LayoutBuffer(Col(4, 4), shape=(4, 4))

        def transpose_body(ctx):
            gid = KernelBuilder.global_id_1d(ctx)
            val = ctx.load_flat(0, gid)
            # Inverse the Row layout to get (i, j), then store with Col layout
            i, j = ctx.apply_inverse(Row(4, 4), gid)
            ctx.store(val, 1, [i, j])

        builder = KernelBuilder(buffers=[src, dst], kernel_body=transpose_body, name="transpose")
        words, _ = compile_to_spirv(builder)
        assert words[0] == 0x07230203

    @requires_spirv
    def test_shared_memory_buffer(self):
        """Kernel with shared memory buffer."""
        from lego.backend.spirv import compile_to_spirv

        src = LayoutBuffer(Row(64), shape=(64,))
        dst = LayoutBuffer(Row(64), shape=(64,))
        smem = LayoutBuffer(Row(64), shape=(64,), shared=True)

        def smem_kernel(ctx):
            gid = KernelBuilder.global_id_1d(ctx)
            val = ctx.load_flat(0, gid)
            ctx.store_flat(val, 2, gid)  # write to shared
            ctx.barrier()
            val2 = ctx.load_flat(2, gid)  # read from shared
            ctx.store_flat(val2, 1, gid)

        builder = KernelBuilder(
            buffers=[src, dst, smem], kernel_body=smem_kernel,
            name="smem_test", grid=(1, 1, 1), block=(64, 1, 1),
        )
        ctx_m, module = builder.build_module()
        mlir = str(module)
        assert "gpu.barrier" in mlir


# ══════════════════════════════════════════════════════════════════════════
# Tier 6: Unified lego.compile() API
# ══════════════════════════════════════════════════════════════════════════

class TestUnifiedCompile:
    """Test the top-level lego.compile() dispatcher."""

    @requires_spirv
    def test_compile_cpu(self):
        import lego
        compiler = lego.compile(Row(64), shape=(64,), target="cpu")
        # CPU returns a LayoutCompiler
        assert hasattr(compiler, "transform_numpy")

    @requires_spirv
    @requires_naga
    def test_compile_webgpu(self):
        import lego
        result = lego.compile(Row(64), shape=(64,), target="webgpu")
        assert isinstance(result, lego.CompileResult)
        assert "@compute" in result.kernel_source

    @requires_spirv
    @requires_ptxas
    def test_compile_cuda_produces_llvm_ir(self):
        """CUDA target goes through lego-to-llvm, produces LLVM IR."""
        import lego
        result = lego.compile(Row(64), shape=(64,), target="cuda")
        assert isinstance(result, lego.CompileResult)
        assert result.target == "cuda"
        # LLVM IR should contain the function
        assert "llvm." in result.kernel_source or "define" in result.kernel_source

    @requires_spirv
    def test_compile_invalid_target(self):
        import lego
        with pytest.raises(ValueError, match="Unknown target"):
            lego.compile(Row(64), shape=(64,), target="opencl")


# ══════════════════════════════════════════════════════════════════════════
# Tier 7: LLVM+SPIR-V pipeline — lego-to-llvmspirv
# ══════════════════════════════════════════════════════════════════════════

class TestLLVMSPIRVPipeline:
    """Test the lego-to-llvmspirv pipeline (GPU→LLVM with SPIR-V calling conventions)."""

    @requires_llvmspirv
    def test_row_layout_compiles(self):
        """Simple Row layout compiles through lego-to-llvmspirv."""
        from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer, make_permutation_kernel
        builder = make_permutation_kernel(Row(64), (64,), "f32", 64)
        result = builder.compile(target="llvmspirv")
        assert result.target == "llvmspirv"
        assert "spir_funccc" in result.kernel_source or "llvm." in result.kernel_source

    @requires_llvmspirv
    def test_col_layout_compiles(self):
        """Col layout compiles through lego-to-llvmspirv."""
        from lego.backend.gpu_builder import make_permutation_kernel
        builder = make_permutation_kernel(Col(8, 8), (8, 8), "f32", 64)
        result = builder.compile(target="llvmspirv")
        assert result.target == "llvmspirv"

    @requires_llvmspirv
    def test_tileby_layout_compiles(self):
        """TileBy layout compiles through lego-to-llvmspirv."""
        from lego.backend.gpu_builder import make_permutation_kernel
        layout = OrderBy(Row(8, 8)).TileBy([4, 4], [2, 2])
        builder = make_permutation_kernel(layout, (8, 8), "f32", 64)
        result = builder.compile(target="llvmspirv")
        assert result.target == "llvmspirv"

    @requires_llvmspirv
    def test_transpose_naive_compiles(self):
        """Transpose-naive-style kernel compiles through lego-to-llvmspirv."""
        N, TD = 64, 16
        A_layout = OrderBy(Row(N, N)).TileBy([N // TD, N // TD], [TD, TD])
        B_layout = OrderBy(Row(N, N)).TileBy([N // TD, N // TD], [TD, TD])
        A = LayoutBuffer(A_layout, shape=(N, N))
        B = LayoutBuffer(B_layout, shape=(N, N))

        def body(ctx):
            gid = KernelBuilder.global_id_1d(ctx)
            val = ctx.load_flat(0, gid)
            ctx.store_flat(val, 1, gid)

        builder = KernelBuilder(buffers=[A, B], kernel_body=body,
                                name="transpose_test", grid=(N*N//256, 1, 1), block=(256, 1, 1))
        result = builder.compile(target="llvmspirv")
        assert result.target == "llvmspirv"

    @requires_llvmspirv
    def test_unified_compile_llvmspirv(self):
        """lego.compile(target='llvmspirv') works via GPUTarget registry."""
        import lego
        result = lego.compile(Row(64), shape=(64,), target="llvmspirv")
        assert isinstance(result, lego.CompileResult)
        assert result.target == "llvmspirv"
