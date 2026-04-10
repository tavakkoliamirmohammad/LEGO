"""Intel GPU (SPIRV/XeVM) backend plugin for lego-layout."""

import ctypes
import pathlib


def register():
    """Load SPIRV plugin and register Intel GPU targets."""
    _lib_dir = pathlib.Path(__file__).parent / "_mlir_libs"
    for suffix in (".so", ".dylib"):
        lib_path = _lib_dir / f"libLegoSPIRVPlugin{suffix}"
        if lib_path.exists():
            break
    else:
        raise RuntimeError(
            "Intel plugin native library not found. "
            "Expected libLegoSPIRVPlugin.{so,dylib} in " + str(_lib_dir)
        )

    _lib = ctypes.CDLL(str(lib_path))
    _lib.legoPluginRegisterSPIRV()

    from lego.backend.gpu_builder import GPUTarget
    GPUTarget(
        name="llvmspirv",
        pipeline="lego-to-llvmspirv",
        default_chip="generic",
        default_format="assembly",
        tmp_prefix="lego_llvmspirv_",
    ).register()

    GPUTarget(
        name="intel",
        pipeline="lego-to-xevm",
        default_chip="bmg",
        default_format="assembly",
        tmp_prefix="lego_intel_",
    ).register()
