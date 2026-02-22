import sys
import z3

def verify(smt2_file):
    try:
        # Load the SMT2 file
        assertions = z3.parse_smt2_file(smt2_file)
        
        # Create a solver
        solver = z3.Solver()
        solver.add(assertions)
        
        # Check satisfiability
        result = solver.check()
        
        if result == z3.sat:
            print("SAT: Out-of-bounds access is possible!")
            model = solver.model()
            print("Counterexample:")
            # Sort declarations by name for deterministic output
            decls = sorted(model.decls(), key=lambda d: d.name())
            for decl in decls:
                print(f"  {decl.name()} = {model[decl]}")
            sys.exit(1)
            
        elif result == z3.unsat:
            print("UNSAT: Bounds are provably safe.")
            sys.exit(0)
            
        else:
            print(f"UNKNOWN: Solver returned {result}")
            sys.exit(2)
            
    except Exception as e:
        print(f"ERROR executing Z3 solver on {smt2_file}: {e}")
        sys.exit(3)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_bounds.py <model.smt2>")
        sys.exit(1)
    verify(sys.argv[1])
