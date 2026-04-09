---
name: run-tests
description: Run the LEGO test suite (lit + pytest)
---

Run the LEGO test suites in order:

1. **MLIR lit tests + Python tests**: `ninja -C build check-lego-all`
2. If the above target is unavailable, fall back to running separately:
   - `lit -v test/`
   - `python -m pytest python/tests/ -v`

Report results from both. If either fails, show the failing test details with context.
