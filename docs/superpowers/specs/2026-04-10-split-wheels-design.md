# Split Wheels: Additive GPU Backend Plugins for LEGO

**Date:** 2026-04-10
**Status:** Approved design, implementation not started
**Branch:** `feature/split-wheels`

---

## Problem

LEGO ships a single monolithic wheel (`lego-layout`) containing:
- All Python code (~100 KB)
- `libLegoPythonCAPI.{so,dylib}` (~172 MB uncompressed) embedding MLIR/LLVM core + all GPU codegen backends

This causes two concrete problems:

1. **PyPI's 100 MB file-size limit.** PR #72 had to drop AMDGPU from CI wheels to fit. ROCm users cannot install LEGO from PyPI — they must build from source.
2. **Users download backends they don't need.** A Triton developer on NVIDIA hardware downloads SPIRV and (formerly) AMDGPU codegen they'll never use.

### Size breakdown (from `build/llvm-build/lib/` static archives)

| Component | Uncompressed | Notes |
|-----------|-------------|-------|
| MLIR/LLVM core + host target | ~71 MB | Always needed |
| NVPTX/NVVM (CUDA) | ~14 MB | |
| AMDGPU/ROCDL (ROCm) | ~53 MB | Currently dropped from wheels |
| SPIRV/XeVM (Intel) | ~34 MB | Includes MLIR SPIR-V dialect |
| **Total** | **~172 MB** | |

---

## Solution: Additive Plugin Wheels via dlopen

Split into four PyPI packages. GPU backends are separate shared libraries that register into the base at runtime via `dlopen`. Zero duplication of MLIR/LLVM core.

### Package structure

```
lego-layout              (~25 MB compressed wheel)
  lego/                  <- all Python code
  lego/mlir/_mlir_libs/  <- core CAPI lib (MLIR + LEGO dialect + host target only)

lego-layout-cuda         (~5 MB compressed)
  lego_cuda/
  lego_cuda/_mlir_libs/libLegoNVPTXPlugin.{so,dylib}

lego-layout-rocm         (~18 MB compressed)
  lego_rocm/
  lego_rocm/_mlir_libs/libLegoAMDGPUPlugin.{so,dylib}

lego-layout-intel        (~12 MB compressed)
  lego_intel/
  lego_intel/_mlir_libs/libLegoSPIRVPlugin.{so,dylib}
```

### User install experience

```bash
pip install lego-layout              # CPU JIT, all frontends, all source-gen
pip install "lego-layout[cuda]"      # + CUDA GPU compilation
pip install "lego-layout[rocm]"      # + ROCm (now possible again!)
pip install "lego-layout[cuda,rocm]" # both simultaneously
pip install "lego-layout[all]"       # everything
```

### Estimated wheel sizes

| Package | Uncompressed | Compressed (est.) |
|---------|-------------|-------------------|
| `lego-layout` | ~71 MB | ~25 MB |
| `lego-layout-cuda` | ~14 MB | ~5 MB |
| `lego-layout-rocm` | ~53 MB | ~18 MB |
| `lego-layout-intel` | ~34 MB | ~12 MB |
| **All combined** | **~172 MB** | **~60 MB** |

All individual packages well under PyPI's 100 MB limit.

---

## How Plugin Loading Works

### Build time

- Core `libLegoPythonCAPI.{so,dylib}` built with `LEGO_LLVM_TARGETS=X86` (Linux) or `AArch64` (macOS). No GPU codegen linked.
- Each plugin `.so` is a separate shared library that statically links only its target's LLVM codegen + MLIR dialect libs. It dynamically resolves MLIR/LLVM core symbols against the already-loaded CAPI library.

### Runtime loading chain

```
import lego
  -> lego/backend/dialects/lego_dialect.py calls register_lego_dialect()
    -> _legoDialects.so loads libLegoPythonCAPI via RTLD_GLOBAL
       (MLIR's _mlir_libs/__init__.py already does this)
  -> lego/backend/gpu_builder.py at module load:
    -> calls _load_backend_plugins()
    -> uses importlib.metadata.entry_points(group="lego.backends")
    -> finds installed plugins (e.g. lego_cuda, lego_rocm)
    -> each plugin's entry point:
       1. ctypes.CDLL("libLegoNVPTXPlugin.so") - symbols resolve against core
       2. calls legoRegisterNVPTXPlugin() via ctypes - registers pipeline + translations
       3. GPUTarget("cuda", "lego-to-nvvm", ...).register() - Python-level registry
```

### Why RTLD_GLOBAL makes this work

MLIR's upstream `_mlir_libs/__init__.py` loads `libLegoPythonCAPI` with `RTLD_GLOBAL`, making all MLIR/LLVM symbols globally visible. When a plugin `.so` is loaded via `ctypes.CDLL`, its unresolved MLIR symbols bind against the already-loaded core. No special linker flags or `RPATH` tricks needed beyond what MLIR already provides.

LLVM target registration is global and additive — `LLVMInitializeNVPTXTarget()` static constructors run at `dlopen` time and register into LLVM's global `TargetRegistry`. Multiple targets coexist. MLIR pass registration works the same way.

### Error handling

When a user calls `lego.compile(layout, target="cuda")` without `lego-layout-cuda` installed, `"cuda"` is not in `_GPU_TARGETS`. The existing `compile()` error path fires with an improved message:

```
ValueError: Unknown target 'cuda'. Installed: cpu, vulkan, webgpu, metal, webgl.
Install CUDA backend with: pip install "lego-layout[cuda]"
```

---

## C++ Changes

### Split registration functions

**Current state:** `legoRegisterPasses()` in `lib/Lego/CAPI/Dialects.cpp:101-129` registers everything in one call. `registerLegoPipelines()` in `lib/Lego/Passes.cpp:209-251` registers all pipelines behind `#ifdef` guards.

**New state:**

`legoRegisterPasses()` becomes `legoRegisterCorePasses()`:
- Registers canonicalize, CSE, arith passes
- Registers all 9 LEGO-specific passes (LegoToArith, LegoNormalization, etc.)
- Registers always-available pipelines: `lego-lower`, `lego-to-llvm`, `lego-to-spirv`
- Called from `lego-opt.cpp` and `LegoDialectPybind.cpp`

Three new `extern "C"` plugin registration functions:
- `legoRegisterNVPTXPlugin()` — registers `lego-to-nvvm` pipeline + NVVM translation interfaces
- `legoRegisterAMDGPUPlugin()` — registers `lego-to-rocdl` pipeline + ROCDL translation interfaces
- `legoRegisterSPIRVPlugin()` — registers `lego-to-llvmspirv` + `lego-to-xevm` pipelines + XeVM translation interfaces

### New plugin source files

```
lib/Lego/plugin/
  CMakeLists.txt
  NVPTXPlugin.cpp    # wraps buildLegoToNVVMPipeline(), exports legoRegisterNVPTXPlugin()
  AMDGPUPlugin.cpp   # wraps buildLegoToROCDLPipeline(), exports legoRegisterAMDGPUPlugin()
  SPIRVPlugin.cpp    # wraps buildLegoToLLVMSPIRVPipeline() + buildLegoToXeVMPipeline(),
                     #   exports legoRegisterSPIRVPlugin()
```

Each plugin file is thin — it calls the existing pipeline builders from `LegoNVVMPipeline.cpp`, etc. No logic duplication.

### New CMake targets

```cmake
# lib/Lego/plugin/CMakeLists.txt

if(LEGO_HAS_NVPTX AND MLIR_ENABLE_BINDINGS_PYTHON)
  add_library(LegoNVPTXPlugin SHARED NVPTXPlugin.cpp)
  target_link_libraries(LegoNVPTXPlugin PRIVATE
    LegoPythonCAPI
    MLIRNVVMTarget MLIRNVVMDialect MLIRNVVMToLLVM
    MLIRGPUToNVVMTransforms MLIRNVVMToLLVMIRTranslation
    MLIRGPUToLLVMIRTranslation)
  set_target_properties(LegoNVPTXPlugin PROPERTIES
    OUTPUT_NAME "LegoNVPTXPlugin"
    SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

# same pattern for AMDGPU, SPIRV/XeVM
```

### Changes to existing CMake

`lib/Lego/CMakeLists.txt` — the `MLIRLego` library drops GPU-backend-specific libs from its `LINK_LIBS` (lines 160-161: `${_GPU_BACKEND_LIBS}`). These move to the plugin targets. The `#ifdef`-guarded pipeline `.cpp` files still compile as part of `MLIRLego` (they're needed for `lego-opt`) but the plugin `.so` files are what ship in the plugin wheels.

**Important:** `lego-opt` continues to link everything monolithically. It calls both `legoRegisterCorePasses()` and all plugin registration functions (guarded by `#ifdef`). Only the Python wheel path splits.

### Changes to CAPI header

`include/Lego/CAPI/Dialects.h` — rename `legoRegisterPasses()` to `legoRegisterCorePasses()`. Add declarations for the three plugin functions (guarded by `#ifdef`).

---

## Python Changes

### `python/lego/backend/gpu_builder.py`

Remove hardcoded `GPUTarget(...).register()` calls (lines 180-211). Replace with entry-point discovery:

```python
def _load_backend_plugins():
    """Discover and load GPU backend plugins via entry points."""
    from importlib.metadata import entry_points
    for ep in entry_points(group="lego.backends"):
        try:
            register_fn = ep.load()
            register_fn()
        except Exception:
            pass  # plugin installed but native lib missing — skip silently

_load_backend_plugins()
```

### `python/lego/__init__.py`

Improve the error message in `compile()` to suggest installing plugins. No structural changes needed — dispatch via `_GPU_TARGETS` dict already works.

### Plugin Python packages

Each plugin is a minimal Python package with an `__init__.py` that loads the native lib and registers the GPU target.

**`lego_cuda/__init__.py`:**
```python
import ctypes
import pathlib

def register():
    """Load NVPTX plugin and register CUDA GPU target."""
    _lib_dir = pathlib.Path(__file__).parent / "_mlir_libs"
    # Find the plugin shared library
    for suffix in (".so", ".dylib"):
        lib_path = _lib_dir / f"libLegoNVPTXPlugin{suffix}"
        if lib_path.exists():
            break
    else:
        raise RuntimeError("CUDA plugin native library not found")

    _lib = ctypes.CDLL(str(lib_path))
    _lib.legoRegisterNVPTXPlugin()

    from lego.backend.gpu_builder import GPUTarget
    GPUTarget(
        name="cuda",
        pipeline="lego-to-nvvm",
        default_chip=None,
        default_format="fatbin",
        tmp_prefix="lego_cuda_",
    ).register()
```

Same pattern for `lego_rocm/__init__.py` and `lego_intel/__init__.py`.

### Plugin `pyproject.toml` (one per plugin)

**`python/plugins/cuda/pyproject.toml`:**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "lego-layout-cuda"
version = "0.2.2"
description = "CUDA (NVPTX) backend plugin for lego-layout"
requires-python = ">=3.12"
dependencies = ["lego-layout>=0.2.2"]

[project.entry-points."lego.backends"]
cuda = "lego_cuda:register"
```

### Base `python/pyproject.toml` additions

```toml
[project.optional-dependencies]
cuda = ["lego-layout-cuda>=0.2.2"]
rocm = ["lego-layout-rocm>=0.2.2"]
intel = ["lego-layout-intel>=0.2.2"]
all = [
    "lego-layout-cuda>=0.2.2",
    "lego-layout-rocm>=0.2.2",
    "lego-layout-intel>=0.2.2",
]
```

---

## CI Changes

### Build matrix

Add a `target` dimension to the build matrix:

```yaml
matrix:
  os: [ubuntu-22.04, macos-15]
  python: ["3.12", "3.13", "3.14"]
  target: [core, cuda, rocm, intel]
```

Each target variant builds with different `LEGO_LLVM_TARGETS`:

| `target` | Linux `LEGO_LLVM_TARGETS` | macOS `LEGO_LLVM_TARGETS` |
|----------|--------------------------|--------------------------|
| `core` | `X86` | `AArch64` |
| `cuda` | `X86;NVPTX` | `AArch64;NVPTX` |
| `rocm` | `X86;AMDGPU` | `AArch64;AMDGPU` |
| `intel` | `X86;SPIRV` | `AArch64;SPIRV` |

### Wheel assembly

- `core` builds: existing `lego-prepare-wheel` target, produces `lego-layout` wheel
- Plugin builds: new `lego-prepare-plugin-wheel` CMake target that copies only the plugin `.so` + thin Python package into a separate staging directory

### Verification

```bash
# Core-only verification
pip install lego-layout --find-links dist/
python -c "import lego; lego.compile(lego.Tiled([4,4], [2,2]), shape=(4,4), target='cpu')"

# Plugin verification (per target)
pip install lego-layout-cuda --find-links dist/
python -c "from lego.backend.gpu_builder import _GPU_TARGETS; assert 'cuda' in _GPU_TARGETS"
```

### Publishing

All four packages publish to PyPI via the same trusted publishing action. Each package needs its own PyPI project configured with trusted publishing.

---

## What Does NOT Change

- **All Python APIs** — `lego.jit`, `lego.compile`, `lego.Tiled`, all frontends, all printers
- **`lego-opt` CLI** — stays monolithic, links everything statically
- **LEGO dialect C++** — `LegoDialect.cpp`, `LegoOps.cpp`, all passes, all verification
- **GPU pipeline `.cpp` files** — `LegoNVVMPipeline.cpp`, etc. keep their `#ifdef` guards and logic
- **WASM build** — completely separate path
- **Test infrastructure** — `check-lego-all` runs against monolithic `lego-opt`. Python tests run with all plugins installed
- **`lego.mlir` namespace** — stays under `lego.mlir.*`, no namespace package issues

---

## File Change Summary

| Layer | Files Modified | Files Created |
|-------|---------------|---------------|
| C++ registration | `lib/Lego/CAPI/Dialects.cpp`, `lib/Lego/CAPI/Dialects.h` (rename function), `lib/Lego/Passes.cpp` (split pipeline registration) | `lib/Lego/plugin/NVPTXPlugin.cpp`, `lib/Lego/plugin/AMDGPUPlugin.cpp`, `lib/Lego/plugin/SPIRVPlugin.cpp` |
| CMake | `lib/Lego/CMakeLists.txt` (drop GPU libs from MLIRLego link), root `CMakeLists.txt` (add plugin wheel targets) | `lib/Lego/plugin/CMakeLists.txt` |
| Python (base) | `python/lego/backend/gpu_builder.py` (entry-point discovery), `python/lego/__init__.py` (better error msg), `python/pyproject.toml` (extras) | |
| Python (plugins) | | `python/plugins/cuda/{pyproject.toml,setup_wheel.py,lego_cuda/__init__.py}`, same for `rocm/`, `intel/` |
| CI | `.github/workflows/ci.yml` (target matrix + plugin wheel steps) | |
| Tests | `python/check_imports.py` (update) | Plugin verification script |

**~10 files modified, ~15 files created. No deletions of existing logic.**

---

## Risks and Mitigations

### Symbol visibility across dlopen (Linux)

MLIR's `_mlir_libs/__init__.py` already loads `libLegoPythonCAPI` with `RTLD_GLOBAL`. Plugin `.so` files resolve symbols through this mechanism. If a platform has issues, `ctypes.CDLL(path, ctypes.RTLD_GLOBAL)` can be used explicitly. macOS flat namespaces handle this natively.

**Mitigation:** Test on both Linux (manylinux) and macOS in CI.

### CI build time (4x matrix)

Each plugin build currently rebuilds MLIR/LLVM core. This can be optimized later by caching the core build artifact and only compiling the plugin delta.

**Mitigation:** Start with simple 4x matrix. Optimize with build caching if CI time becomes a problem.

### Version coupling between base and plugins

A plugin built against `lego-layout` 0.2.2 may not work with 0.2.3 if the CAPI ABI changes.

**Mitigation:** Pin plugin dependencies to exact base version (`lego-layout==0.2.2`) or use compatible-release specifiers (`lego-layout~=0.2.2`). Bump all four packages together on every release.

### auditwheel / delocate for plugin wheels

`auditwheel repair` on Linux may try to bundle MLIR core symbols into the plugin wheel (since the plugin `.so` has unresolved symbols at build time).

**Mitigation:** Use `auditwheel repair --exclude libLegoPythonCAPI.so` to tell it those symbols come from the base package. Similar `--exclude` for `delocate-wheel` on macOS.
