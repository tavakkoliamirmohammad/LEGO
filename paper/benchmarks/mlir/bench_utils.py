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


def detect_nvvm_target():
    """Detect GPU compute capability and return (chip, features) for lego-to-nvvm.

    Falls back to sm_80 / +ptx78 which works with CUDA 12+ and most modern GPUs.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            # e.g. "8.6" → "sm_86"
            cap = r.stdout.strip().split("\n")[0].strip()
            chip = "sm_" + cap.replace(".", "")
            # Pick a PTX version that matches the arch generation
            major = int(cap.split(".")[0])
            ptx = {7: "+ptx70", 8: "+ptx78", 9: "+ptx80"}.get(major, "+ptx78")
            return chip, ptx
    except Exception:
        pass
    return "sm_80", "+ptx78"


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


def _make_main_wrapper(builder, init_mod=10000):
    """Generate a @main function that allocates, initializes, calls the kernel,
    and prints the last (output) buffer in flat order.

    Args:
        init_mod: Modulus for input buffer initialization.
            val[i] = (i * 37 + 17) % init_mod.
            Default 10000 for transpose/permutation tests.
            Use smaller values (e.g. 10) for accumulation kernels (matmul,
            reduce) so that results stay within f32 exact integer range.
    """
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
            # val = (i * 37 + 17) % init_mod — deterministic, non-sequential,
            # exact in f32, makes permutation bugs obvious.
            lines.append(f"    %c37_{i} = arith.constant 37 : i32")
            lines.append(f"    %c17_{i} = arith.constant 17 : i32")
            lines.append(f"    %c{init_mod}_{i} = arith.constant {init_mod} : i32")
            lines.append(f"    scf.for %i{i} = %c0 to %n{i} step %c1 {{")
            lines.append(f"      %vi{i} = arith.index_cast %i{i} : index to i32")
            lines.append(f"      %t{i} = arith.muli %vi{i}, %c37_{i} : i32")
            lines.append(f"      %t2_{i} = arith.addi %t{i}, %c17_{i} : i32")
            lines.append(f"      %ri{i} = arith.remui %t2_{i}, %c{init_mod}_{i} : i32")
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
    chip, features = detect_nvvm_target()
    for target in targets:
        try:
            kwargs = {"name": f"{builder._name}_{target}"}
            if target == "cuda":
                kwargs["chip"] = chip
                kwargs["features"] = features
            result = builder.compile(target=target, **kwargs)
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
    for t in ("vulkan", "webgpu", "metal"):
        if t in targets:
            run_gpu_verify(builder, expected, t, label=f"{N}x{N}")


def run_cuda_verify(builder, expected, label=None, atol=0, rtol=0, init_mod=10000):
    """Run the builder's generated kernel on GPU and verify against expected output.

    Builds the actual LEGO kernel, injects a @main wrapper, lowers through
    lego-to-nvvm via the Python PassManager API (no lego-opt binary needed),
    and executes via mlir-runner.

    Args:
        atol: Absolute tolerance. 0 = exact integer comparison,
              >0 = np.allclose on float values.
        rtol: Relative tolerance (used with atol when atol>0).
        init_mod: Modulus for input init (must match _init_data).

    Returns True/False/None (None = skipped).
    """
    mlir_runner, libs_dir = find_mlir_runner()
    if mlir_runner is None:
        print("  CUDA execution: SKIP (mlir-runner not found)", file=sys.stderr)
        return None

    from lego.mlir.ir import Context, Location, Module
    from lego.mlir.passmanager import PassManager
    from lego.backend.dialects.lego_dialect import register as register_lego

    # Build the kernel module and get MLIR text
    build_ctx, build_mod = builder.build_module()
    with build_ctx:
        mlir_str = str(build_mod)

    # Inject @main wrapper before the closing }
    main_code = _make_main_wrapper(builder, init_mod=init_mod)
    idx = mlir_str.rstrip().rfind("}")
    mlir_src = mlir_str[:idx] + main_code + "\n}\n"

    # Lower via Python PassManager (replaces lego-opt subprocess)
    ctx = Context()
    register_lego(ctx)
    ctx.load_all_available_dialects()
    try:
        with ctx, Location.unknown():
            module = Module.parse(mlir_src)
            chip, features = detect_nvvm_target()
            pm = PassManager.parse(
                f"builtin.module(lego-to-nvvm{{chip={chip} features={features}}})"
            )
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
            raw = [float(v.strip()) for v in match.group(1).split(",")]
            if atol > 0 or rtol > 0:
                gpu_values = np.array(raw, dtype=np.float32)
                expected_f = expected.astype(np.float32) if hasattr(expected, 'astype') else np.array(expected, dtype=np.float32)
                ok = np.allclose(gpu_values, expected_f, atol=atol, rtol=rtol)
                max_err = float(np.max(np.abs(gpu_values - expected_f)))
            else:
                gpu_values = np.array([int(v) for v in raw])
                ok = np.array_equal(gpu_values, expected)
                max_err = int(np.max(np.abs(gpu_values - expected)))
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


def _init_data(n, init_mod=10000):
    """Deterministic pseudo-random integers as f32: (i * 37 + 17) % init_mod."""
    return ((np.arange(n, dtype=np.int32) * 37 + 17) % init_mod).astype(np.float32)



def _make_binding_array(kind, val):
    """Convert a single binding descriptor to a numpy array."""
    if kind == "const_int":
        return np.array([val], dtype=np.uint32)
    elif kind == "const_float":
        return np.array([val], dtype=np.float32)
    else:
        raise ValueError(f"Unknown binding kind: {kind}")


def _prepare_data_arrays(builder, init_mod=10000):
    """Create input/output numpy arrays for a kernel's global buffers.

    Skips shared (workgroup) buffers — those are device-side only.
    Returns (arrays_dict, last_data_index) where last_data_index is the
    output buffer's index in _global_bufs.
    """
    global_bufs = builder._global_bufs
    # Find the last non-shared buffer (the output buffer)
    last = max(j for j, g in enumerate(global_bufs) if not g.shared)
    arrays = {}
    for j, gbuf in enumerate(global_bufs):
        if gbuf.shared:
            continue
        if j < last:
            arrays[j] = _init_data(gbuf.numel, init_mod=init_mod)
        else:
            arrays[j] = np.zeros(gbuf.numel, dtype=np.float32)
    return arrays, last


def _compare_output(out_data, expected, atol, rtol):
    """Compare GPU output to expected. Returns (ok, max_err)."""
    if atol > 0 or rtol > 0:
        expected_f = expected.astype(np.float32) if hasattr(expected, 'astype') \
            else np.array(expected, dtype=np.float32)
        ok = np.allclose(out_data, expected_f, atol=atol, rtol=rtol)
        max_err = float(np.max(np.abs(out_data - expected_f)))
    else:
        gpu_int = out_data.astype(np.int32)
        ok = np.array_equal(gpu_int, expected)
        max_err = int(np.max(np.abs(gpu_int - expected)))
    return ok, max_err


def _dispatch_wgpu(builder, shader_code, entry_point, metadata, init_mod):
    """Dispatch via wgpu (Vulkan or WebGPU). Returns (out_data, gpu_info) or raises."""
    import wgpu

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()

    shader = device.create_shader_module(code=shader_code)
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": shader, "entry_point": entry_point})

    bindings = metadata["bindings"]
    grid = metadata["grid"]
    data_arrays, last = _prepare_data_arrays(builder, init_mod)

    # Build wgpu buffers (data + constants interleaved per binding order)
    data_bufs = {}
    for j in data_arrays:
        if j < last:
            data_bufs[j] = device.create_buffer_with_data(
                data=data_arrays[j], usage=wgpu.BufferUsage.STORAGE)
        else:
            data_bufs[j] = device.create_buffer(
                size=data_arrays[j].nbytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    entries = []
    binding_idx = 0
    for kind, val in bindings:
        if kind == "workgroup":
            # Workgroup memory is device-side only — no host binding.
            continue
        if kind == "data":
            buf = data_bufs[val]
        else:
            buf = device.create_buffer_with_data(
                data=_make_binding_array(kind, val), usage=wgpu.BufferUsage.STORAGE)
        entries.append({"binding": binding_idx, "resource": {"buffer": buf}})
        binding_idx += 1

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0), entries=entries)

    enc = device.create_command_encoder()
    p = enc.begin_compute_pass()
    p.set_pipeline(pipeline)
    p.set_bind_group(0, bind_group)
    p.dispatch_workgroups(*grid)
    p.end()
    device.queue.submit([enc.finish()])

    out_data = np.frombuffer(
        device.queue.read_buffer(data_bufs[last]).cast("f"), dtype=np.float32)
    gpu_info = f"{adapter.info['device']}, {adapter.info['backend_type']}"
    return out_data, gpu_info


def _dispatch_metal(builder, metal_source, entry_point, metadata, init_mod):
    """Dispatch via PyObjC Metal. Returns (out_data, gpu_info) or raises."""
    import Metal

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("no Metal device")

    library, err = device.newLibraryWithSource_options_error_(
        metal_source, Metal.MTLCompileOptions.new(), None)
    if library is None:
        raise RuntimeError(f"Metal compile: {err}")

    func = library.newFunctionWithName_(entry_point)
    if func is None:
        raise RuntimeError(f"function '{entry_point}' not found")

    pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
    if pipeline is None:
        raise RuntimeError(f"pipeline: {err}")

    bindings = metadata["bindings"]
    grid = metadata["grid"]
    threads = metadata["threads"]
    data_arrays, last = _prepare_data_arrays(builder, init_mod)

    # Create Metal buffers per binding order (skip workgroup — device-side only)
    metal_bufs = []
    for kind, val in bindings:
        if kind == "workgroup":
            continue
        if kind == "data":
            arr = data_arrays[val]
        else:
            arr = _make_binding_array(kind, val)
        metal_bufs.append(device.newBufferWithBytes_length_options_(
            arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared))

    queue = device.newCommandQueue()
    cmd_buf = queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)

    # Set threadgroup memory for workgroup buffers (before data buffers).
    tg_idx = 0
    for kind, val in bindings:
        if kind == "workgroup":
            wg_size = val * 4  # val = element count, 4 bytes per f32
            encoder.setThreadgroupMemoryLength_atIndex_(wg_size, tg_idx)
            tg_idx += 1

    # Bind data/constant buffers after the threadgroup slots.
    for i, buf in enumerate(metal_bufs):
        encoder.setBuffer_offset_atIndex_(buf, 0, tg_idx + i)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(*grid), Metal.MTLSizeMake(*threads))
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()
    if cmd_buf.error() is not None:
        raise RuntimeError(f"GPU error: {cmd_buf.error()}")

    # Find output buffer (index into metal_bufs which skips workgroup entries)
    buf_idx = 0
    out_buf = None
    for k, v in bindings:
        if k == "workgroup":
            continue
        if k == "data" and v == last:
            out_buf = metal_bufs[buf_idx]
            break
        buf_idx += 1
    out_data = np.frombuffer(
        out_buf.contents().as_buffer(out_buf.length()), dtype=np.float32).copy()
    return out_data, device.name()


def run_gpu_verify(builder, expected, target, label=None, atol=0, rtol=0,
                   init_mod=10000):
    """Run a compiled kernel on GPU and verify correctness.

    Supports targets: "vulkan", "webgpu", "metal".
    Returns True/False/None (None = skipped).
    """
    target_name = target.capitalize()

    # --- Compile ---
    try:
        if target == "vulkan":
            from lego.backend.spirv import compile_to_spirv
            spv_words, _, metadata = compile_to_spirv(builder)
            shader_code = np.array(spv_words, dtype=np.uint32).tobytes()
            entry_point = metadata["entry_point"]
        else:
            from lego.backend.spirv import compile_to_target
            result = compile_to_target(builder, target=target, name=f"verify_{target}")
            metadata = result.metadata
            shader_code = result.kernel_source
            if target == "webgpu":
                ep = re.search(r'fn (\w+)\(@builtin', shader_code)
                entry_point = ep.group(1) if ep else metadata["entry_point"]
            elif target == "metal":
                shader_code = _fix_metal_buffer_bindings(shader_code)
                ep = re.search(r'kernel void (\w+)\(', shader_code)
                entry_point = ep.group(1) if ep else metadata["entry_point"]
    except Exception as e:
        print(f"  {target_name} execution: FAIL (compile: {e})", file=sys.stderr)
        return False

    # --- Dispatch ---
    try:
        if target in ("vulkan", "webgpu"):
            try:
                import wgpu  # noqa: F401
            except ImportError:
                print(f"  {target_name} execution: SKIP (wgpu not installed)",
                      file=sys.stderr)
                return None
            out_data, gpu_info = _dispatch_wgpu(
                builder, shader_code, entry_point, metadata, init_mod)
        elif target == "metal":
            try:
                import Metal  # noqa: F401
            except ImportError:
                print("  Metal execution: SKIP (pyobjc-framework-Metal not installed)",
                      file=sys.stderr)
                return None
            out_data, gpu_info = _dispatch_metal(
                builder, shader_code, entry_point, metadata, init_mod)
        else:
            print(f"  {target_name} execution: SKIP (unknown target)", file=sys.stderr)
            return None
    except Exception as e:
        print(f"  {target_name} execution: FAIL ({e})", file=sys.stderr)
        return False

    # --- Verify ---
    ok, max_err = _compare_output(out_data, expected, atol, rtol)
    tag = f" ({label})" if label else ""
    print(f"  {target_name} execution{tag}: {'PASS' if ok else 'FAIL'} "
          f"— {len(expected)} elements, max error={max_err} [{gpu_info}]",
          file=sys.stderr)
    return ok


def run_benchmark(builder, compute_expected_fn, targets, label=None, atol=0, rtol=0,
                   init_mod=10000):
    """Generic benchmark runner: compile to targets and verify correctness.

    Args:
        builder: KernelBuilder instance.
        compute_expected_fn: callable (inputs: list[np.ndarray]) -> np.ndarray.
            Given the list of input arrays (deterministic init pattern), returns
            the expected flat output array.
        targets: list of compilation target strings.
        label: optional label for verification output.
        atol: absolute tolerance (0 = exact integer match, >0 = float tolerance).
    """
    from lego.backend.gpu_builder import _ensure_stack_size
    _ensure_stack_size()

    # Generate MLIR
    mlir_ctx, module = builder.build_module()
    print(module)

    # Compile to GPU backends
    chip, features = detect_nvvm_target()
    for target in targets:
        try:
            kwargs = {"name": f"{builder._name}_{target}"}
            if target == "cuda":
                kwargs["chip"] = chip
                kwargs["features"] = features
            result = builder.compile(target=target, **kwargs)
            print(f"\n--- {target}: {result.kernel_path} ---", file=sys.stderr)
        except Exception as e:
            print(f"\n--- {target}: FAILED ({e}) ---", file=sys.stderr)

    # Compute expected output from deterministic input data
    global_bufs = builder._global_bufs
    inputs = []
    for buf in global_bufs[:-1]:
        inputs.append(_init_data(buf.numel, init_mod=init_mod))
    expected = compute_expected_fn(inputs)

    # Host-side verification
    print("\nVerification:", file=sys.stderr)
    run_cuda_verify(builder, expected, label=label, atol=atol, rtol=rtol, init_mod=init_mod)
    for t in ("vulkan", "webgpu", "metal"):
        if t in targets:
            run_gpu_verify(builder, expected, t, label=label, atol=atol, rtol=rtol,
                           init_mod=init_mod)


def _fix_metal_buffer_bindings(metal_source):
    """Replace naga's [[user(fake0)]] with proper [[buffer(N)]] attributes.

    Accounts for threadgroup parameters in the kernel signature, which
    occupy implicit argument table slots before the [[buffer(N)]] entries.
    """
    # Count threadgroup parameters in the kernel signature (they occupy
    # argument table indices before the explicit buffer bindings).
    kernel_match = re.search(r'kernel void \w+\([^)]+\)', metal_source, re.DOTALL)
    threadgroup_count = 0
    if kernel_match:
        sig = kernel_match.group(0)
        threadgroup_count = len(re.findall(r'threadgroup\s+\w+', sig))

    idx = threadgroup_count
    def repl(m):
        nonlocal idx
        result = f"[[buffer({idx})]]"
        idx += 1
        return result
    return re.sub(r'\[\[user\(fake0\)\]\]', repl, metal_source)


def run_metal_verify(builder, expected, label=None, atol=0, rtol=0, init_mod=10000):
    """Run via SPIR-V → naga → Metal Shading Language → Metal GPU.

    Uses PyObjC to dispatch the compute kernel on Apple GPU.
    Returns True/False/None (None = skipped).
    """
    try:
        import Metal
    except ImportError:
        print("  Metal execution: SKIP (pyobjc-framework-Metal not installed)",
              file=sys.stderr)
        return None

    from lego.backend.spirv import compile_to_target

    try:
        result = compile_to_target(builder, target="metal", name="verify_metal")
    except Exception as e:
        print(f"  Metal execution: FAIL (compile: {e})", file=sys.stderr)
        return False

    bindings = result.metadata["bindings"]
    grid = result.metadata["grid"]
    threads = result.metadata["threads"]

    # Fix naga's [[user(fake0)]] → [[buffer(N)]]
    metal_source = _fix_metal_buffer_bindings(result.kernel_source)

    # Find entry point name
    ep_match = re.search(r'kernel void (\w+)\(', metal_source)
    if not ep_match:
        print("  Metal execution: FAIL (no kernel entry point in Metal source)",
              file=sys.stderr)
        return False
    entry_point = ep_match.group(1)

    try:
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            print("  Metal execution: SKIP (no Metal device)", file=sys.stderr)
            return None

        # Compile Metal source
        options = Metal.MTLCompileOptions.new()
        library, err = device.newLibraryWithSource_options_error_(
            metal_source, options, None)
        if library is None:
            print(f"  Metal execution: FAIL (compile: {err})", file=sys.stderr)
            return False

        func = library.newFunctionWithName_(entry_point)
        if func is None:
            print(f"  Metal execution: FAIL (function '{entry_point}' not found)",
                  file=sys.stderr)
            return False

        pipeline, err = device.newComputePipelineStateWithFunction_error_(func, None)
        if pipeline is None:
            print(f"  Metal execution: FAIL (pipeline: {err})", file=sys.stderr)
            return False

        # Create buffers matching SPIR-V binding order
        global_bufs = builder._global_bufs
        last_data = len(global_bufs) - 1
        data_arrays = {}
        for j, gbuf in enumerate(global_bufs):
            n = gbuf.numel
            if j < last_data:
                data_arrays[j] = _init_data(n, init_mod=init_mod)
            else:
                data_arrays[j] = np.zeros(n, dtype=np.float32)

        metal_bufs = []
        for kind, val in bindings:
            if kind == "const_int":
                arr = np.array([val], dtype=np.uint32)
            elif kind == "const_float":
                arr = np.array([val], dtype=np.float32)
            elif kind == "const":
                arr = np.array([val], dtype=np.uint32)
            else:
                arr = data_arrays[val]
            buf = device.newBufferWithBytes_length_options_(
                arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared)
            metal_bufs.append(buf)

        # Dispatch
        queue = device.newCommandQueue()
        cmd_buf = queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)

        for i, buf in enumerate(metal_bufs):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        threads_per_grid = Metal.MTLSizeMake(grid[0], grid[1], grid[2])
        threads_per_tg = Metal.MTLSizeMake(threads[0], threads[1], threads[2])
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            threads_per_grid, threads_per_tg)
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        if cmd_buf.error() is not None:
            print(f"  Metal execution: FAIL (GPU error: {cmd_buf.error()})",
                  file=sys.stderr)
            return False

        # Read back output buffer
        out_buf = None
        for kind, val in bindings:
            if kind == "data" and val == last_data:
                out_buf = metal_bufs[bindings.index((kind, val))]
                break

        if out_buf is None:
            print("  Metal execution: FAIL (output buffer not found)", file=sys.stderr)
            return False

        out_ptr = out_buf.contents().as_buffer(out_buf.length())
        out_data = np.frombuffer(out_ptr, dtype=np.float32).copy()

        if atol > 0 or rtol > 0:
            ok = np.allclose(out_data, expected.astype(np.float32), atol=atol, rtol=rtol)
            max_err = float(np.max(np.abs(out_data - expected.astype(np.float32))))
        else:
            gpu_values = out_data.astype(np.int32)
            ok = np.array_equal(gpu_values, expected)
            max_err = int(np.max(np.abs(gpu_values - expected)))
        tag = f" ({label})" if label else ""
        print(f"  Metal execution{tag}: {'PASS' if ok else 'FAIL'} "
              f"— {len(expected)} elements, max error={max_err} [{device.name()}]",
              file=sys.stderr)
        return ok
    except Exception as e:
        print(f"  Metal execution: FAIL ({e})", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False
