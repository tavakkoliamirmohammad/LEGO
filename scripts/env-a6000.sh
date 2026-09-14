# Source this on the CHPC A6000 node before running lego_synth GPU harness tests.
export CUDA_HOME=/uufs/chpc.utah.edu/sys/spack/v10/linux-rocky8-x86_64/none-none/cuda-12.9.0-juabbspist2v5gugt3uw45mbs2dyksec
export GCC_ROOT=/uufs/chpc.utah.edu/sys/spack/v10/linux-rocky8-nehalem/gcc-8.5.0/gcc-13.4.0-r5tlvf4ek2zw3zuofly3smlbfhggisrh
export NINJA_DIR=/uufs/chpc.utah.edu/sys/spack/v10/linux-rocky8-nehalem/gcc-8.5.0/ninja-1.12.1-xgdi76lgsuetjud7hxmqf663lkrhkt4d/bin
export PATH=$GCC_ROOT/bin:$CUDA_HOME/bin:$NINJA_DIR:$PATH
export LD_LIBRARY_PATH=$GCC_ROOT/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export CC=$GCC_ROOT/bin/gcc CXX=$GCC_ROOT/bin/g++
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/scratch/general/vast/u1419116/torch_ext}
export MLIR_BUILD_DIR=/scratch/general/vast/u1419116/LEGO/build
