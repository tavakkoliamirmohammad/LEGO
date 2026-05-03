"""Measure 18_tblis_notranspose.

CASTLE candidate 10. Layout class: TBLIS.
Prior verdicts: AMD WIN, Intel LOSS.

XFAIL: XFAIL pending R18: TBLIS runtime microkernel selection not expressible in cpu_dsl v1
"""
import json
import math
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

XFAIL_REASON = "XFAIL pending R18: TBLIS runtime microkernel selection not expressible in cpu_dsl v1"
N_BENCH = 16384

PRIOR_VERDICT_AMD = "WIN"
PRIOR_VERDICT_INTEL = "LOSS"


def main():
    print(json.dumps({
        "name": "18_tblis_notranspose",
        "N": N_BENCH,
        "layout_class": "TBLIS",
        "prior_verdict": "LOSS",
        "numpy_ms": float('nan'),
        "scalar_jit_ms": float('nan'),
        "vec_jit_ms": float('nan'),
        "speedup_isolated_jit": float('nan'),
        "speedup_vs_numpy": float('nan'),
        "verdict": "SKIP",
        "notes": "SKIP — " + XFAIL_REASON,
    }))


if __name__ == "__main__":
    main()
