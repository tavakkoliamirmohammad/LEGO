"""Shared utilities for LEGO GPU transpose benchmarks."""

import os
import re
import subprocess
import sys
import tempfile
import numpy as np


def verify_layouts(layouts, N):
    """Verify layouts are valid bijective permutations.

    Args:
        layouts: dict mapping name to layout object (e.g., {"A": A_layout, "B": B_layout})
        N: matrix dimension (NxN)

    Returns True if all layouts are bijective.
    """
    from lego.backend.compiler import DType, LayoutCompiler

    all_ok = True
    parts = []
    for name, layout in layouts.items():
        compiler = LayoutCompiler(layout, (N, N), DType.f32)
        fwd, inv = compiler.get_permutation_table()
        ok = len(np.unique(fwd)) == N * N and np.all(inv[fwd] == np.arange(N * N))
        parts.append(f"{name} bijective={ok}")
        if not ok:
            all_ok = False

    status = "PASS" if all_ok else "FAIL"
    print(f"  Layout verification ({N}x{N}): {status} ({', '.join(parts)})",
          file=sys.stderr)
    return all_ok


def find_mlir_runner():
    """Locate mlir-runner and its shared libs directory.

    Set MLIR_BUILD_DIR to the LLVM/MLIR build directory containing
    bin/mlir-runner and lib/libmlir_*_runtime.so, e.g.:
        export MLIR_BUILD_DIR=/path/to/llvm-project/build

    Returns (mlir_runner_path, libs_dir) or (None, None) if not found.
    """
    build_dir = os.environ.get("MLIR_BUILD_DIR")
    if not build_dir:
        print("  CUDA execution: SKIP (set MLIR_BUILD_DIR to your LLVM build dir)",
              file=sys.stderr)
        return None, None
    mlir_runner = os.path.join(build_dir, "bin/mlir-runner")
    libs_dir = os.path.join(build_dir, "lib")
    if not os.path.isfile(mlir_runner):
        print(f"  CUDA execution: SKIP (mlir-runner not found in {build_dir}/bin)",
              file=sys.stderr)
        return None, None
    return mlir_runner, libs_dir


def _make_main_wrapper(builder):
    """Generate a @main function that allocates, initializes, calls the kernel,
    and prints the last (output) buffer in flat order."""
    global_bufs = builder._global_bufs
    name = builder._name

    lines = []
    lines.append(f"  func.func @main() {{")
    lines.append(f"    %c0 = arith.constant 0 : index")
    lines.append(f"    %c1 = arith.constant 1 : index")

    buf_names = []
    memref_types = []
    for i, buf in enumerate(global_bufs):
        n = buf.numel
        ty = f"memref<{n}xf32>"
        bname = f"%buf{i}"
        buf_names.append(bname)
        memref_types.append(ty)

        lines.append(f"    %n{i} = arith.constant {n} : index")
        lines.append(f"    {bname} = memref.alloc() : {ty}")

        if i < len(global_bufs) - 1:
            # Input buffers: pseudo-random integers as f32.
            # val = (i * 37 + 17) % 10000 — deterministic, non-sequential,
            # exact in f32, makes permutation bugs obvious.
            lines.append(f"    %c37_{i} = arith.constant 37 : i32")
            lines.append(f"    %c17_{i} = arith.constant 17 : i32")
            lines.append(f"    %c10000_{i} = arith.constant 10000 : i32")
            lines.append(f"    scf.for %i{i} = %c0 to %n{i} step %c1 {{")
            lines.append(f"      %vi{i} = arith.index_cast %i{i} : index to i32")
            lines.append(f"      %t{i} = arith.muli %vi{i}, %c37_{i} : i32")
            lines.append(f"      %t2_{i} = arith.addi %t{i}, %c17_{i} : i32")
            lines.append(f"      %ri{i} = arith.remui %t2_{i}, %c10000_{i} : i32")
            lines.append(f"      %vf{i} = arith.sitofp %ri{i} : i32 to f32")
            lines.append(f"      memref.store %vf{i}, {bname}[%i{i}] : {ty}")
            lines.append(f"    }}")
        else:
            # Output buffer: zeros
            lines.append(f"    %zero = arith.constant 0.0 : f32")
            lines.append(f"    scf.for %i{i} = %c0 to %n{i} step %c1 {{")
            lines.append(f"      memref.store %zero, {bname}[%i{i}] : {ty}")
            lines.append(f"    }}")

    # gpu.host_register all buffers
    for i in range(len(global_bufs)):
        lines.append(f"    %u{i} = memref.cast {buf_names[i]} : {memref_types[i]} to memref<*xf32>")
        lines.append(f"    gpu.host_register %u{i} : memref<*xf32>")

    # Call the actual generated kernel
    args = ", ".join(buf_names)
    sig = ", ".join(memref_types)
    lines.append(f"    func.call @{name}({args}) : ({sig}) -> ()")

    # Print last buffer flat (output verification)
    last = len(global_bufs) - 1
    lines.append(f"    func.call @printMemrefF32(%u{last}) : (memref<*xf32>) -> ()")

    # Dealloc
    for i in range(len(global_bufs)):
        lines.append(f"    memref.dealloc {buf_names[i]} : {memref_types[i]}")

    lines.append(f"    return")
    lines.append(f"  }}")
    lines.append(f"  func.func private @printMemrefF32(memref<*xf32>)")
    return "\n".join(lines)


def run_transpose_benchmark(builder, layouts, N, targets, extra_verify=None):
    """Shared main logic for transpose benchmarks.

    Args:
        builder: KernelBuilder instance
        layouts: dict of layout name -> layout object for bijectivity check
        N: matrix dimension (NxN)
        targets: list of compilation targets (e.g., ["cuda", "webgpu", "metal"])
        extra_verify: optional callable for additional verification (e.g., WebGPU)
    """
    from lego.backend.gpu_builder import _ensure_stack_size
    _ensure_stack_size()

    # Generate MLIR
    mlir_ctx, module = builder.build_module()
    print(module)

    # Compile to GPU backends
    for target in targets:
        try:
            result = builder.compile(target=target, name=f"{builder._name}_{target}")
            print(f"\n--- {target}: {result.kernel_path} ---", file=sys.stderr)
        except Exception as e:
            print(f"\n--- {target}: FAILED ({e}) ---", file=sys.stderr)

    # Host-side verification
    print("\nVerification:", file=sys.stderr)
    verify_layouts(layouts, N)
    # Init values match the MLIR wrapper: (i * 37 + 17) % 10000
    init = (np.arange(N * N, dtype=np.int32) * 37 + 17) % 10000
    expected = init.reshape(N, N).T.ravel()
    run_cuda_verify(builder, expected, label=f"{N}x{N}")
    if "vulkan" in targets:
        run_vulkan_verify(builder, expected, label=f"{N}x{N}")
    if "webgpu" in targets:
        run_webgpu_verify(builder, expected, label=f"{N}x{N}")


def run_cuda_verify(builder, expected, label=None):
    """Run the builder's generated kernel on GPU and verify against expected output.

    Builds the actual LEGO kernel, injects a @main wrapper, lowers through
    lego-to-nvvm via the Python PassManager API (no lego-opt binary needed),
    and executes via mlir-runner.

    Returns True/False/None (None = skipped).
    """
    mlir_runner, libs_dir = find_mlir_runner()
    if mlir_runner is None:
        print("  CUDA execution: SKIP (mlir-runner not found)", file=sys.stderr)
        return None

    from mlir.ir import Context, Location, Module
    from mlir.passmanager import PassManager
    from lego.backend.dialects.lego_dialect import register as register_lego

    # Build the kernel module and get MLIR text
    build_ctx, build_mod = builder.build_module()
    with build_ctx:
        mlir_str = str(build_mod)

    # Inject @main wrapper before the closing }
    main_code = _make_main_wrapper(builder)
    idx = mlir_str.rstrip().rfind("}")
    mlir_src = mlir_str[:idx] + main_code + "\n}\n"

    # Lower via Python PassManager (replaces lego-opt subprocess)
    ctx = Context()
    register_lego(ctx)
    ctx.load_all_available_dialects()
    try:
        with ctx, Location.unknown():
            module = Module.parse(mlir_src)
            pm = PassManager.parse("builtin.module(lego-to-nvvm)")
            pm.run(module.operation)
        lowered_ir = str(module)
    except Exception as e:
        print(f"  CUDA execution: FAIL (lego-to-nvvm: {e})", file=sys.stderr)
        return False

    # Write lowered IR and run mlir-runner
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(lowered_ir)
        lowered_path = f.name

    try:
        env = dict(os.environ,
                   LD_LIBRARY_PATH=libs_dir + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        r = subprocess.run(
            [mlir_runner, lowered_path,
             f"--shared-libs={libs_dir}/libmlir_cuda_runtime.so,"
             f"{libs_dir}/libmlir_c_runner_utils.so,"
             f"{libs_dir}/libmlir_runner_utils.so",
             "--entry-point-result=void"],
            capture_output=True, text=True, env=env, timeout=60)

        match = re.search(r'data\s*=\s*\n\[([^\]]+)\]', r.stdout)
        if match:
            gpu_values = np.array([int(float(v.strip())) for v in match.group(1).split(",")])
            ok = np.array_equal(gpu_values, expected)
            max_err = np.max(np.abs(gpu_values - expected))
            tag = f" ({label})" if label else ""
            print(f"  CUDA execution{tag}: {'PASS' if ok else 'FAIL'} "
                  f"— {len(expected)} elements, max error={max_err}",
                  file=sys.stderr)
            return ok
        else:
            print(f"  CUDA execution: FAIL (no output: {r.stdout[:200]})",
                  file=sys.stderr)
            return False
    except Exception as e:
        print(f"  CUDA execution: FAIL ({e})", file=sys.stderr)
        return False
    finally:
        os.unlink(lowered_path)


def _init_data(n):
    """Deterministic pseudo-random integers as f32: (i * 37 + 17) % 10000."""
    return ((np.arange(n, dtype=np.int32) * 37 + 17) % 10000).astype(np.float32)


def _parse_spirv_launch(mlir):
    """Parse bindings and grid dimensions from lowered SPIR-V MLIR.

    Returns (bindings, grid) where bindings is a list of ("const", value)
    or ("data", buf_index) in the order they appear in gpu.launch_func args,
    and grid is [x, y, z] workgroup counts.
    """
    bindings = []
    args_match = re.search(r'args\((.+)\)', mlir)
    if args_match:
        buf_idx = 0
        for m in re.finditer(r'(%c(\d+)\s*:\s*index|%arg\d+\s*:\s*memref)',
                             args_match.group(1)):
            if m.group(0).startswith('%c'):
                bindings.append(("const", int(m.group(2))))
            else:
                bindings.append(("data", buf_idx))
                buf_idx += 1

    grid = [1, 1, 1]
    blocks_match = re.search(r'blocks in \(([^)]+)\)', mlir)
    if blocks_match:
        for i, part in enumerate(blocks_match.group(1).split(',')):
            m = re.match(r'\s*%c(\d+)', part.strip())
            if m:
                grid[i] = int(m.group(1))

    return bindings, grid


def _run_wgpu_dispatch(builder, expected, label, target_name,
                       shader_code, entry_point, bindings, grid):
    """Shared wgpu dispatch: create buffers, run kernel, verify output."""
    import wgpu

    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        device = adapter.request_device_sync()
    except Exception as e:
        print(f"  {target_name} execution: SKIP (no GPU adapter: {e})",
              file=sys.stderr)
        return None

    try:
        shader = device.create_shader_module(code=shader_code)
        pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": shader, "entry_point": entry_point})

        # Allocate data buffers (input: pseudo-random, output: zeros+readable)
        global_bufs = builder._global_bufs
        last_data = len(global_bufs) - 1
        data_bufs = {}
        for j, gbuf in enumerate(global_bufs):
            n = gbuf.numel
            if j < last_data:
                data_bufs[j] = device.create_buffer_with_data(
                    data=_init_data(n), usage=wgpu.BufferUsage.STORAGE)
            else:
                data_bufs[j] = device.create_buffer(
                    size=n * 4,
                    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

        # Build binding entries in MLIR arg order (interleaved consts + data)
        entries = []
        for i, (kind, val) in enumerate(bindings):
            if kind == "const":
                buf = device.create_buffer_with_data(
                    data=np.array([val], dtype=np.uint32),
                    usage=wgpu.BufferUsage.STORAGE)
            else:
                buf = data_bufs[val]
            entries.append({"binding": i, "resource": {"buffer": buf}})

        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0), entries=entries)

        enc = device.create_command_encoder()
        p = enc.begin_compute_pass()
        p.set_pipeline(pipeline)
        p.set_bind_group(0, bind_group)
        p.dispatch_workgroups(*grid)
        p.end()
        device.queue.submit([enc.finish()])

        # Read back and compare
        out_data = np.frombuffer(
            device.queue.read_buffer(data_bufs[last_data]).cast("f"),
            dtype=np.float32)
        gpu_values = out_data.astype(np.int32)
        ok = np.array_equal(gpu_values, expected)
        max_err = int(np.max(np.abs(gpu_values - expected)))
        tag = f" ({label})" if label else ""
        gpu_info = f"{adapter.info['device']}, {adapter.info['backend_type']}"
        print(f"  {target_name} execution{tag}: {'PASS' if ok else 'FAIL'} "
              f"— {len(expected)} elements, max error={max_err} [{gpu_info}]",
              file=sys.stderr)
        return ok
    except Exception as e:
        print(f"  {target_name} execution: FAIL ({e})", file=sys.stderr)
        return False


def run_webgpu_verify(builder, expected, label=None):
    """Run via SPIR-V → naga → WGSL → wgpu. Returns True/False/None."""
    try:
        import wgpu  # noqa: F401
    except ImportError:
        print("  WebGPU execution: SKIP (wgpu not installed)", file=sys.stderr)
        return None

    from lego.backend.spirv import compile_to_target, compile_to_spirv

    try:
        result = compile_to_target(builder, target="webgpu", name="verify")
    except Exception as e:
        print(f"  WebGPU execution: FAIL (compile: {e})", file=sys.stderr)
        return False

    _, mlir = compile_to_spirv(builder)
    bindings, grid = _parse_spirv_launch(mlir)

    ep_match = re.search(r'fn (\w+)\(@builtin', result.kernel_source)
    if not ep_match:
        print("  WebGPU execution: FAIL (no entry point in WGSL)", file=sys.stderr)
        return False

    return _run_wgpu_dispatch(
        builder, expected, label, "WebGPU",
        result.kernel_source, ep_match.group(1), bindings, grid)


def run_vulkan_verify(builder, expected, label=None):
    """Run via raw SPIR-V binary → wgpu (Vulkan). Returns True/False/None."""
    try:
        import wgpu  # noqa: F401
    except ImportError:
        print("  Vulkan execution: SKIP (wgpu not installed)", file=sys.stderr)
        return None

    from lego.backend.spirv import compile_to_spirv

    try:
        spv_words, mlir = compile_to_spirv(builder)
    except Exception as e:
        print(f"  Vulkan execution: FAIL (compile: {e})", file=sys.stderr)
        return False

    bindings, grid = _parse_spirv_launch(mlir)
    spv_bytes = np.array(spv_words, dtype=np.uint32).tobytes()

    # Entry point name from gpu.launch_func @module::@entry
    ep_match = re.search(r'gpu\.launch_func\s+@\w+::@(\w+)', mlir)
    if not ep_match:
        print("  Vulkan execution: FAIL (no entry point in MLIR)", file=sys.stderr)
        return False

    return _run_wgpu_dispatch(
        builder, expected, label, "Vulkan",
        spv_bytes, ep_match.group(1), bindings, grid)
