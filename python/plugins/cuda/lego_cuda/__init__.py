"""CUDA (NVPTX) backend plugin for lego-layout."""

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
        raise RuntimeError(
            "CUDA plugin native library not found. "
            "Expected libLegoNVPTXPlugin.{so,dylib} in " + str(_lib_dir)
        )

    _lib = ctypes.CDLL(str(lib_path))
    _lib.legoPluginRegisterNVPTX()

    from lego.backend.gpu_builder import GPUTarget
    GPUTarget(
        name="cuda",
        pipeline="lego-to-nvvm",
        default_chip=None,
        default_format="fatbin",
        default_features=None,
        tmp_prefix="lego_cuda_",
    ).register()
