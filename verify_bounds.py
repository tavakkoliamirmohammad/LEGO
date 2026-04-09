import sys
import subprocess

def verify(smt2_file):
    try:
        # Run z3 executable
        result = subprocess.run(['z3', smt2_file], capture_output=True, text=True)
        
        output = result.stdout.strip()
        if output.startswith('sat'):
            print("SAT: Out-of-bounds access is possible!")
            # Note: We could run z3 again with (get-model) if needed
            sys.exit(1)
        elif output.startswith('unsat'):
            print("UNSAT: Bounds are provably safe.")
            sys.exit(0)
        else:
            print(f"UNKNOWN/ERROR: Solver returned:\n{output}\nStderr: {result.stderr}")
            sys.exit(2)
            
    except Exception as e:
        print(f"ERROR executing Z3 solver on {smt2_file}: {e}")
        sys.exit(3)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_bounds.py <model.smt2>")
        sys.exit(1)
    verify(sys.argv[1])
