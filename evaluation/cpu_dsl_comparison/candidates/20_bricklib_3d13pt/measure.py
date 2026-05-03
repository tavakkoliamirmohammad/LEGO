"""Measure 20_bricklib_3d13pt.

CASTLE candidate 12. Layout class: Brick.
Prior verdicts: AMD PARITY, Intel WIN.

XFAIL: XFAIL pending R12: brick stride not threaded through; BrickLib not bundled
"""
import json
import math
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

XFAIL_REASON = "XFAIL pending R12: brick stride not threaded through; BrickLib not bundled"
N_BENCH = 16384

PRIOR_VERDICT_AMD = "PARITY"
PRIOR_VERDICT_INTEL = "WIN"


def main():
    print(json.dumps({
        "name": "20_bricklib_3d13pt",
        "N": N_BENCH,
        "layout_class": "Brick",
        "prior_verdict": "WIN",
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
