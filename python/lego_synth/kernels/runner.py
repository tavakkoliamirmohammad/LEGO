"""Run a KernelBuilder kernel on the GPU through lego-to-nvvm and mlir-runner with f16 inputs.

Inputs are initialised on the host deterministically as small integers exactly
representable in f16 (((i*37+17) % mod) - mod/2), so f32 accumulation is exact and
results can be compared bit-for-bit against NumPy.
"""
import os
import re
import subprocess
import tempfile

import numpy as np


def init_values(n: int, mod: int = 16) -> np.ndarray:
    return (((np.arange(n, dtype=np.int64) * 37 + 17) % mod) - mod // 2).astype(np.float32)


def _main_wrapper(builder, mod: int) -> str:
    bufs = builder._global_bufs
    lines = ["  func.func @main() {", "    %c0 = arith.constant 0 : index", "    %c1 = arith.constant 1 : index",
             f"    %cmod = arith.constant {mod} : i32", f"    %chalf = arith.constant {mod // 2} : i32",
             "    %c37 = arith.constant 37 : i32", "    %c17 = arith.constant 17 : i32"]
    names, types = [], []

    def dt(b):
        return getattr(b.dtype, "value", str(b.dtype))

    for i, b in enumerate(bufs):
        ty = f"memref<{b.numel}x{dt(b)}>"
        names.append(f"%buf{i}"); types.append(ty)
        lines += [f"    %n{i} = arith.constant {b.numel} : index", f"    %buf{i} = memref.alloc() : {ty}",
                  f"    scf.for %i{i} = %c0 to %n{i} step %c1 {{"]
        if i < len(bufs) - 1:
            lines += [f"      %vi{i} = arith.index_cast %i{i} : index to i32",
                      f"      %t{i} = arith.muli %vi{i}, %c37 : i32", f"      %u{i} = arith.addi %t{i}, %c17 : i32",
                      f"      %m{i} = arith.remui %u{i}, %cmod : i32", f"      %s{i} = arith.subi %m{i}, %chalf : i32",
                      f"      %f{i} = arith.sitofp %s{i} : i32 to f32"]
            if dt(b) == "f16":
                lines += [f"      %h{i} = arith.truncf %f{i} : f32 to f16", f"      memref.store %h{i}, %buf{i}[%i{i}] : {ty}"]
            else:
                lines += [f"      memref.store %f{i}, %buf{i}[%i{i}] : {ty}"]
        else:
            lines += [f"      %z{i} = arith.constant 0.0 : {dt(b)}", f"      memref.store %z{i}, %buf{i}[%i{i}] : {ty}"]
        lines += ["    }", f"    %cast{i} = memref.cast %buf{i} : {ty} to memref<*x{dt(b)}>",
                  f"    gpu.host_register %cast{i} : memref<*x{dt(b)}>"]
    last = len(bufs) - 1
    lines += [f"    func.call @{builder._name}({', '.join(names)}) : ({', '.join(types)}) -> ()",
              f"    func.call @printMemrefF32(%cast{last}) : (memref<*xf32>) -> ()"]
    lines += [f"    memref.dealloc {n} : {t}" for n, t in zip(names, types)]
    lines += ["    return", "  }", "  func.func private @printMemrefF32(memref<*xf32>)"]
    return "\n".join(lines)


def run_kernel(builder, mod: int = 16, chip: str = "sm_86", features: str = "+ptx78", timeout: int = 120):
    """Compile with lego-to-nvvm, execute with mlir-runner, return the output buffer as np.float32."""
    from lego.mlir.ir import Context, Module
    from lego.mlir.passmanager import PassManager
    from lego.backend.dialects.lego_dialect import register as register_lego
    build_dir = os.environ.get("MLIR_BUILD_DIR", "")
    runner, libs = os.path.join(build_dir, "bin", "mlir-runner"), os.path.join(build_dir, "lib")
    if not os.path.isfile(runner):
        raise RuntimeError("MLIR_BUILD_DIR must contain bin/mlir-runner")
    bctx, bmod = builder.build_module()
    with bctx:
        src = str(bmod)
    src = src[:src.rstrip().rfind("}")] + _main_wrapper(builder, mod) + "\n}\n"
    ctx = Context(); register_lego(ctx); ctx.load_all_available_dialects()
    with ctx:
        m = Module.parse(src)
        PassManager.parse(f"builtin.module(lego-to-nvvm{{chip={chip} features={features}}})").run(m.operation)
        lowered = str(m)
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(lowered); path = f.name
    try:
        env = dict(os.environ, LD_LIBRARY_PATH=libs + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        r = subprocess.run([runner, path, f"--shared-libs={libs}/libmlir_cuda_runtime.so,{libs}/libmlir_c_runner_utils.so,{libs}/libmlir_runner_utils.so",
                            "--entry-point-result=void"], capture_output=True, text=True, env=env, timeout=timeout)
        mt = re.search(r"data\s*=\s*\n\[([^\]]+)\]", r.stdout)
        if not mt:
            raise RuntimeError(f"no output from mlir-runner: rc={r.returncode}\n{r.stderr[-2000:]}")
        return np.array([float(v) for v in mt.group(1).split(",")], dtype=np.float32)
    finally:
        os.unlink(path)
