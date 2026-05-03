# 37_stencil_nonpow2_brick

**CASTLE candidate:** 29 — BrickLib stencil with non-pow-2 brick size (severe LOSS)
**Layout class:** Brick+non-pow2
**Prior verdicts:** AMD LOSS, Intel LOSS

## XFAIL

XFAIL pending R12: brick stride not threaded through; BrickLib not bundled

## Kernel

BrickLib stencil with non-pow-2 brick size (severe LOSS)

## Expected behavior

SKIP — see XFAIL above.
