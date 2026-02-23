import argparse
import os
import sys

# Define base paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRITON_DIR = os.path.join(SCRIPT_DIR, "triton")
CUDA_DIR = os.path.join(SCRIPT_DIR, "cuda")
MLIR_DIR = os.path.join(SCRIPT_DIR, "mlir")

def run_script(script_path, *args):
    """Executes a python script using the current python executable."""
    print(f"Running {os.path.basename(script_path)}...")
    cmd = f'"{sys.executable}" "{script_path}" {" ".join(args)}'
    result = os.system(cmd)
    if result != 0:
        print(f"Error: Command failed with exit code {result}")
        sys.exit(result)

def generate_triton():
    print("=== Generating Triton Kernels ===")
    os.chdir(TRITON_DIR)
    # The new triton sympy scripts will simply be python scripts 
    # that print the @lego.jit source code.
    os.system(f'"{sys.executable}" matmul_sympy.py --NTA --NTB > ../../generated/triton/matmul/matmul_NTA_NTB.py')
    os.system(f'"{sys.executable}" matmul_sympy.py --NTA --TB > ../../generated/triton/matmul/matmul_NTA_TB.py')
    os.system(f'"{sys.executable}" matmul_sympy.py --TA --NTB > ../../generated/triton/matmul/matmul_TA_NTB.py')
    os.system(f'"{sys.executable}" matmul_sympy.py --TA --TB > ../../generated/triton/matmul/matmul_TA_TB.py')
    
    os.system(f'"{sys.executable}" softmax_sympy.py > ../../generated/triton/softmax/softmax.py')
    os.system(f'"{sys.executable}" layernorm_sympy.py > ../../generated/triton/layernorm/layernorm.py')
    os.system(f'"{sys.executable}" grouped_gemm_sympy.py > ../../generated/triton/grouped_gemm/grouped_gemm.py')
    os.chdir(SCRIPT_DIR)

import subprocess

def generate_cuda():
    print("=== Generating CUDA Kernels ===")
    os.chdir(CUDA_DIR)
    
    # Generate directly into the paper/generated/cuda folder
    env = os.environ.copy()
    
    with open('../../generated/cuda/nw/needle_kernel.cu', 'w') as f:
        subprocess.run([sys.executable, "nw_sympy.py"], env=env, stdout=f, check=True)
        
    with open('../../generated/cuda/lud/lud_kernel.cu', 'w') as f:
        subprocess.run([sys.executable, "lud.py"], env=env, stdout=f, check=True)
        
    with open('../../generated/cuda/bricks-laplace/main.cu', 'w') as f:
        subprocess.run([sys.executable, "bricks_laplasian.py"], env=env, stdout=f, check=True)
        
    with open('../../generated/cuda/bricks/main.cu', 'w') as f:
        subprocess.run([sys.executable, "bricks_f3d.py"], env=env, stdout=f, check=True)
    
    os.chdir(SCRIPT_DIR)

def generate_mlir():
    print("=== Generating MLIR Kernels ===")
    os.chdir(MLIR_DIR)
    # Replicate the MLIR generation
    target_dir = "../../generated/mlir/transpose"
    def gen_mlir(script, suffix, *args):
        os.system(f'"{sys.executable}" {script} {" ".join(args)} > {target_dir}/transpose_{suffix}.mlir')

    tests = [
        ("128", "128"), ("256", "256"), ("512", "512"),
        ("1024", "1024"), ("2048", "2048"), ("4096", "4096"), ("8192", "8192")
    ]
    for nx, ny in tests:
        gen_mlir("transpose_naive.py", f"naive_{nx}_{ny}", nx, ny)
        gen_mlir("transpose_smem.py", f"smem_{nx}_{ny}", nx, ny)

    os.chdir(SCRIPT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized kernel generator for LEGO.")
    parser.add_argument("--backend", type=str, choices=["all", "triton", "cuda", "mlir"], 
                        default="all", help="Which backend to generate kernels for.")
    args = parser.parse_args()

    if args.backend in ["all", "triton"]:
        generate_triton()
    if args.backend in ["all", "cuda"]:
        generate_cuda()
    if args.backend in ["all", "mlir"]:
        generate_mlir()
        
    print("=== All Generation Completed ===")
