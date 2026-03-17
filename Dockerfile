# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Build LLVM/MLIR + LEGO (monolithic)
# ──────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION=3.12
ARG NPROC=32

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git lld ccache \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    python3-pip libssl-dev ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Python deps needed at build time (nanobind for bindings, lit for tests)
RUN pip3 install --no-cache-dir nanobind==2.12.0 lit==18.1.8 pytest==9.0.2

WORKDIR /workspace/lego

# Copy submodule first (changes rarely, maximizes layer cache)
COPY third_party/llvm-project third_party/llvm-project

# Copy build system and source
COPY CMakeLists.txt requirements.txt README.md ./
COPY include/ include/
COPY lib/ lib/
COPY tools/ tools/
COPY test/ test/
COPY python/ python/

# Configure: monolithic build with lld, ccache, Release mode
RUN cmake -G Ninja -S . -B build \
      -DLEGO_MONOLITHIC_LLVM=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DLEGO_LLVM_TARGETS="X86;NVPTX" \
      -DLEGO_ENABLE_RUNNERS=ON \
      -DLLVM_USE_LINKER=lld \
      -DLEGO_ENABLE_CCACHE=ON \
      -DCMAKE_INSTALL_PREFIX=/opt/lego

# Build everything
RUN cmake --build build -j${NPROC} --target check-lego-all

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Lightweight runtime image
# ──────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python3-pip \
  && rm -rf /var/lib/apt/lists/*

# Copy built artifacts from builder
COPY --from=builder /workspace/lego/build/python_packages/lego /opt/lego/python_packages/lego
COPY --from=builder /workspace/lego/build/tools/lego-opt/lego-opt /usr/local/bin/lego-opt
COPY --from=builder /workspace/lego/python /opt/lego/python

# Install Python runtime dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Environment
ENV PYTHONPATH=/opt/lego/python_packages/lego:/opt/lego/python
ENV PATH=/usr/local/bin:${PATH}

# Charliecloud bind-mount directories (CHPC: --bind=/uufs --bind=/scratch)
RUN mkdir -p /scratch /uufs

WORKDIR /workspace
CMD ["/bin/bash"]
