"""Naive matrix transpose — unified KernelBuilder API (all GPU backends).

Includes host-side GPU execution verification via both CUDA (mlir-runner)
and WebGPU (wgpu/Vulkan).
"""
import sys
import os
import numpy as np
from lego.core import OrderBy, Row, Col
from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer, _ensure_stack_size

if len(sys.argv) != 3:
    print("Usage: python transpose_naive.py NX NY")
    sys.exit(1)

NX = int(sys.argv[1])
NY = int(sys.argv[2])
N = NX

TILE_DIM = 32
BRX = 8   # BLOCK_ROWS_X
BRY = 32  # BLOCK_ROWS_Y

dimGrid = (NX // TILE_DIM * NY // TILE_DIM, 1, 1)
dimBlock = (BRY * BRX, 1, 1)

# --- Layouts ---
A_layout = OrderBy(Row(N, N)).TileBy(
    [N // TILE_DIM, N // TILE_DIM],
    [TILE_DIM // BRX, TILE_DIM // BRY],
    [BRX, BRY])
B_layout = OrderBy(Row(N, N)).TileBy(
    [N // TILE_DIM, N // TILE_DIM],
    [TILE_DIM // BRY, TILE_DIM // BRX],
    [BRY, BRX])

A = LayoutBuffer(A_layout, shape=(N, N), dtype="f32")
B = LayoutBuffer(B_layout, shape=(N, N), dtype="f32")


def transpose_kernel(ctx):
    bX = ctx.block_id("x")
    tX = ctx.thread_id("x")

    rby, rbx = ctx.apply_inverse(
        OrderBy(Row(N // TILE_DIM, N // TILE_DIM))
        .TileBy([N // TILE_DIM, N // TILE_DIM]), bX)
    wby, wbx = ctx.apply_inverse(
        OrderBy(Col(N // TILE_DIM, N // TILE_DIM))
        .TileBy([N // TILE_DIM, N // TILE_DIM]), bX)

    rty, rtx = ctx.apply_inverse(
        OrderBy(Row(BRX, BRY)).TileBy([BRX, BRY]), tX)
    wty, wtx = ctx.apply_inverse(
        OrderBy(Col(BRY, BRX)).TileBy([BRY, BRX]), tX)

    tile_loop = OrderBy(
        Row(TILE_DIM // BRX, TILE_DIM // BRY)
    ).TileBy([TILE_DIM // BRX, TILE_DIM // BRY])

    def transpose_body(indices, _):
        j, i = indices
        value = ctx.load(0, [rby, rbx, j, i, rty, rtx])
        ctx.store(value, 1, [wby, wbx, i, j, wty, wtx])

    ctx.tile_loop(tile_loop, transpose_body)


builder = KernelBuilder(
    buffers=[A, B],
    kernel_body=transpose_kernel,
    name="transpose_naive",
    grid=dimGrid,
    block=dimBlock,
)


# ============================================================================
# Host verification: layout bijectivity
# ============================================================================

def verify_layouts():
    """Verify both layouts are valid bijective permutations."""
    from lego.backend.compiler import LayoutCompiler

    a_compiler = LayoutCompiler(A_layout, (N, N), "f32")
    b_compiler = LayoutCompiler(B_layout, (N, N), "f32")
    a_fwd, a_inv = a_compiler.get_permutation_table()
    b_fwd, b_inv = b_compiler.get_permutation_table()

    a_ok = len(np.unique(a_fwd)) == N * N and np.all(a_inv[a_fwd] == np.arange(N * N))
    b_ok = len(np.unique(b_fwd)) == N * N and np.all(b_inv[b_fwd] == np.arange(N * N))

    status = "PASS" if (a_ok and b_ok) else "FAIL"
    print(f"  Layout verification ({N}x{N}): {status} "
          f"(A bijective={a_ok}, B bijective={b_ok})", file=sys.stderr)
    return a_ok and b_ok


# ============================================================================
# GPU execution: CUDA via mlir-runner
# ============================================================================

def run_cuda_transpose():
    """Run a small LEGO transpose on the GPU via lego-to-nvvm + mlir-runner."""
    import subprocess, tempfile

    # Use a small 4x4 test — the MLIR test uses lego.row/lego.col directly
    mlir_src = """\
module attributes {gpu.container_module} {
  gpu.module @transpose_module {
    gpu.func @transpose_kernel(%A: memref<16xf32>, %B: memref<16xf32>)
        kernel attributes {known_block_size = array<i32: 16, 1, 1>} {
      %tid = gpu.thread_id x
      %c4_0 = arith.constant 4 : index
      %c4_1 = arith.constant 4 : index
      %row_layout = lego.row [%c4_0, %c4_1] : !lego.layout
      %row, %col = lego.apply_inverse %row_layout(%tid) : !lego.layout -> index, index
      %col_layout = lego.col [%c4_0, %c4_1] : !lego.layout
      %t_idx = lego.apply %col_layout(%row, %col) : !lego.layout
      %val = memref.load %A[%tid] : memref<16xf32>
      memref.store %val, %B[%t_idx] : memref<16xf32>
      gpu.return
    }
  }
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %A = memref.alloc() : memref<16xf32>
    %B = memref.alloc() : memref<16xf32>
    scf.for %i = %c0 to %c16 step %c1 {
      %vi = arith.index_cast %i : index to i32
      %vf = arith.sitofp %vi : i32 to f32
      memref.store %vf, %A[%i] : memref<16xf32>
    }
    %zero = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c16 step %c1 {
      memref.store %zero, %B[%i] : memref<16xf32>
    }
    %Au = memref.cast %A : memref<16xf32> to memref<*xf32>
    %Bu = memref.cast %B : memref<16xf32> to memref<*xf32>
    gpu.host_register %Au : memref<*xf32>
    gpu.host_register %Bu : memref<*xf32>
    gpu.launch_func @transpose_module::@transpose_kernel
        blocks in (%c1, %c1, %c1)
        threads in (%c16, %c1, %c1)
        args(%A : memref<16xf32>, %B : memref<16xf32>)
    func.call @printMemrefF32(%Bu) : (memref<*xf32>) -> ()
    memref.dealloc %A : memref<16xf32>
    memref.dealloc %B : memref<16xf32>
    return
  }
  func.func private @printMemrefF32(memref<*xf32>)
}
"""
    # Find lego-opt and mlir-runner
    build_dir = os.environ.get("LEGO_BUILD_DIR", "")
    lego_opt = os.path.join(build_dir, "tools/lego-opt/lego-opt") if build_dir else "lego-opt"
    # Try common locations
    for candidate in [lego_opt,
                      os.path.join(os.path.dirname(__file__), "../../../build/tools/lego-opt/lego-opt")]:
        if os.path.isfile(candidate):
            lego_opt = candidate
            break

    parent_build = os.environ.get("LEGO_PARENT_BUILD",
                                  "/scratch/general/vast/u1419116/LEGO/build")
    mlir_runner = os.path.join(parent_build, "llvm-build/bin/mlir-runner")
    libs_dir = os.path.join(parent_build, "llvm-build/lib")

    if not os.path.isfile(mlir_runner):
        print("  CUDA execution: SKIP (mlir-runner not found)", file=sys.stderr)
        return None

    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(mlir_src)
        src_path = f.name

    try:
        # Lower
        lowered = src_path + ".lowered"
        r = subprocess.run([lego_opt, src_path, "--lego-to-nvvm", "-o", lowered],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  CUDA execution: FAIL (lego-to-nvvm: {r.stderr[:200]})", file=sys.stderr)
            return False

        # Execute
        env = dict(os.environ, LD_LIBRARY_PATH=libs_dir + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        r = subprocess.run(
            [mlir_runner, lowered,
             f"--shared-libs={libs_dir}/libmlir_cuda_runtime.so,"
             f"{libs_dir}/libmlir_c_runner_utils.so,"
             f"{libs_dir}/libmlir_runner_utils.so",
             "--entry-point-result=void"],
            capture_output=True, text=True, env=env, timeout=30)

        expected = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
        # Parse output: "data = \n[0,  4,  8,  12, ...]"
        import re
        match = re.search(r'data\s*=\s*\n\[([^\]]+)\]', r.stdout)
        if match:
            values = [int(float(v.strip())) for v in match.group(1).split(",")]
            ok = values == expected
            print(f"  CUDA execution (4x4): {'PASS' if ok else 'FAIL'} "
                  f"{'— B = A^T verified' if ok else f'got {values[:8]}...'}", file=sys.stderr)
            return ok
        else:
            print(f"  CUDA execution: FAIL (no output: {r.stdout[:200]})", file=sys.stderr)
            return False
    except Exception as e:
        print(f"  CUDA execution: FAIL ({e})", file=sys.stderr)
        return False
    finally:
        os.unlink(src_path)
        if os.path.exists(src_path + ".lowered"):
            os.unlink(src_path + ".lowered")


# ============================================================================
# GPU execution: WebGPU via wgpu/Vulkan
# ============================================================================

def run_webgpu_transpose():
    """Run a small LEGO transpose on the GPU via lego-to-spirv + wgpu."""
    try:
        import wgpu
    except ImportError:
        print("  WebGPU execution: SKIP (wgpu not installed)", file=sys.stderr)
        return None

    import re
    from lego.backend.spirv import compile_to_target, compile_to_spirv

    # Simple 4x4 transpose kernel
    M = 4
    s = LayoutBuffer(Row(M, M), shape=(M, M))
    d = LayoutBuffer(Row(M, M), shape=(M, M))

    def small_transpose(ctx):
        gid = KernelBuilder.global_id_1d(ctx)
        row, col = ctx.apply_inverse(Row(M, M), gid)
        val = ctx.load(0, [row, col])
        ctx.store(val, 1, [col, row])

    kb = KernelBuilder(buffers=[s, d], kernel_body=small_transpose,
                       name="transpose_test", workgroup_size=16)

    result = compile_to_target(kb, target="webgpu", name="transpose_test")
    wgsl = result.kernel_source

    # Extract constant values from lowered MLIR
    _, mlir = compile_to_spirv(kb)
    match = re.search(r'args\(([^)]+)\)', mlir)
    const_vals = []
    for arg in match.group(1).split(','):
        m = re.match(r'\s*%c(\d+)', arg.strip())
        if m:
            const_vals.append(int(m.group(1)))

    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        device = adapter.request_device_sync()
        shader = device.create_shader_module(code=wgsl)
        ep = re.search(r'fn (\w+)\(@builtin', wgsl).group(1)
        pipeline = device.create_compute_pipeline(layout="auto",
            compute={"module": shader, "entry_point": ep})

        A_data = np.arange(M * M, dtype=np.float32)

        # Build bindings: scalar constants + data arrays
        entries = []
        for i, val in enumerate(const_vals):
            buf = device.create_buffer_with_data(
                data=np.array([val], dtype=np.uint32),
                usage=wgpu.BufferUsage.STORAGE)
            entries.append({"binding": i, "resource": {"buffer": buf}})

        n_const = len(const_vals)
        buf_a = device.create_buffer_with_data(data=A_data, usage=wgpu.BufferUsage.STORAGE)
        buf_b = device.create_buffer(size=M*M*4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        entries.append({"binding": n_const, "resource": {"buffer": buf_a}})
        entries.append({"binding": n_const + 1, "resource": {"buffer": buf_b}})

        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0), entries=entries)

        enc = device.create_command_encoder()
        p = enc.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, bind_group)
        p.dispatch_workgroups(1)
        p.end()
        device.queue.submit([enc.finish()])

        B_data = np.frombuffer(device.queue.read_buffer(buf_b).cast("f"), dtype=np.float32)
        expected = A_data.reshape(M, M).T.ravel()
        ok = np.allclose(B_data, expected)
        print(f"  WebGPU execution (4x4): {'PASS' if ok else 'FAIL'} "
              f"{'— B = A^T verified' if ok else f'got {B_data}'} "
              f"[{adapter.info['device']}, {adapter.info['backend_type']}]",
              file=sys.stderr)
        return ok
    except Exception as e:
        print(f"  WebGPU execution: FAIL ({e})", file=sys.stderr)
        return False


if __name__ == "__main__":
    _ensure_stack_size()

    # Generate MLIR
    mlir_ctx, module = builder.build_module()
    print(module)

    # Compile to all GPU backends
    targets = ["cuda", "webgpu", "metal"]
    for target in targets:
        try:
            result = builder.compile(target=target, name=f"transpose_naive_{target}")
            print(f"\n--- {target}: {result.kernel_path} ---", file=sys.stderr)
        except Exception as e:
            print(f"\n--- {target}: FAILED ({e}) ---", file=sys.stderr)

    # Host-side verification
    print("\nVerification:", file=sys.stderr)
    verify_layouts()
    run_cuda_transpose()
    run_webgpu_transpose()
