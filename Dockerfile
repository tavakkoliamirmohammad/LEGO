# ────────────────
# 1) Builder stage
# ────────────────
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder

ARG LLVM_COMMIT=556ec4a7261447d13703816cd3730a891441e52c
ARG PYTHON_VERSION=3.12

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev python3-pip libssl-dev ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Clone llvm-project (or COPY if you already have it as a git submodule)
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/llvm/llvm-project.git \
    && cd llvm-project \
    && git fetch --depth 1 origin ${LLVM_COMMIT} \
    && git checkout ${LLVM_COMMIT}

RUN pip install pybind11
# Configure & build
WORKDIR /workspace/llvm-project
RUN mkdir build && cd build && \
    cmake -G Ninja ../llvm \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS="mlir" \
      -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
      -DMLIR_ENABLE_CUDA_RUNNER=ON \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DPython3_EXECUTABLE=$(which python${PYTHON_VERSION}) \
      -DCMAKE_INSTALL_PREFIX=/opt/llvm \
      -DLLVM_BUILD_EXAMPLES=OFF
    ninja && ninja install

# ────────────────
# 2) Runtime stage
# ────────────────
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
ARG PYTHON_VERSION=3.12

# copy install from builder
COPY --from=builder /opt/llvm /opt/llvm
COPY --from=builder /workspace/llvm-project/mlir/python/requirements.txt /opt/llvm/mlir/requirements.txt

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends python${PYTHON_VERSION} python3-pip \
  && rm -rf /var/lib/apt/lists/*

# set up environment
ENV LLVM_INSTALL=/opt/llvm
ENV PATH=${LLVM_INSTALL}/bin:${PATH}
ENV LD_LIBRARY_PATH=${LLVM_INSTALL}/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${LLVM_INSTALL}/python_packages/mlir_core

COPY requirements.txt /workspace/
WORKDIR /workspace

COPY requirements.txt .

# 2) install dependencies
RUN pip3 install --no-cache-dir -r /opt/llvm/mlir/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install triton==3.2.0 torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
# 3) now copy the rest of your application code
COPY . .


CMD ["/bin/bash"]
