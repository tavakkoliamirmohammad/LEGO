"""ROCm (AMDGPU) backend plugin for lego-layout."""

import ctypes
import pathlib


def register():
    """Load AMDGPU plugin and register ROCm GPU target."""
    _lib_dir = pathlib.Path(__file__).parent / "_mlir_libs"
    for suffix in (".so", ".dylib"):
        lib_path = _lib_dir / f"libLegoAMDGPUPlugin{suffix}"
        if lib_path.exists():
            break
    else:
        raise RuntimeError(
            "ROCm plugin native library not found. "
            "Expected libLegoAMDGPUPlugin.{so,dylib} in " + str(_lib_dir)
        )

    _lib = ctypes.CDLL(str(lib_path))
    _lib.legoPluginRegisterAMDGPU()

    from lego.backend.gpu_builder import GPUTarget
    GPUTarget(
        name="rocm",
        pipeline="lego-to-rocdl",
        default_chip="gfx900",
        default_format="assembly",
        tmp_prefix="lego_rocm_",
    ).register()
