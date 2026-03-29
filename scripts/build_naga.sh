#!/bin/bash
# Build naga-cli for SPIR-V → WGSL/Metal/GLSL conversion.
# No GPU hardware required.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="${PROJECT_ROOT}/dist/naga"

if command -v naga &> /dev/null; then
    echo "naga already installed: $(which naga)"
    naga --version 2>/dev/null || true
    exit 0
fi

if ! command -v cargo &> /dev/null; then
    echo "Rust not found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

echo "Building naga-cli..."
cargo install naga-cli --root "${INSTALL_DIR}"

echo ""
echo "naga installed at: ${INSTALL_DIR}/bin/naga"
echo "Add to PATH: export PATH=\"${INSTALL_DIR}/bin:\$PATH\""
