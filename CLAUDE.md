# LEGO

MLIR-based compiler dialect for tiled GPU kernels.

## Build

- `cmake -B build -G Ninja -DLEGO_MONOLITHIC_LLVM=ON && ninja -C build`
- WASM: `./build_wasm.sh`

## Test

- All tests: `ninja -C build check-lego-all`
- Python tests: `pytest python/tests/ -v`

## Structure

- `include/Lego/` — TableGen definitions and headers
- `lib/Lego/` — Pass implementations (C++)
- `python/lego/` — Python frontend and printers
- `python/cpp/` — nanobind C++ bindings
- `test/` — MLIR FileCheck tests (lit)
- `tools/lego-opt/` — Standalone optimizer driver
- `third_party/` — Do not edit
- `viz/` — WASM browser visualization
