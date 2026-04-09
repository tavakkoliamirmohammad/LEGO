---
name: build-lego
description: Build the LEGO compiler with correct CMake configuration
---

Build LEGO using ninja (never make):

- **Incremental**: `ninja -C build`
- **Full rebuild**: `cmake -B build -G Ninja -DLEGO_MONOLITHIC_LLVM=ON && ninja -C build`
- **WASM**: `./build_wasm.sh`

Always use the existing `build/` directory for incremental builds. The project uses ccache automatically if available.
