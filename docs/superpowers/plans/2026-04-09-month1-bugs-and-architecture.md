# Month 1: Bug Fixes & Architecture Improvements

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all known correctness bugs and clean up the most impactful architectural debt so the codebase is healthy before adding features.

**Architecture:** Bug fixes are isolated one-file changes. Architecture improvements (A1-A4) refactor duplication across printer and adapter files. Each task produces a working, tested commit.

**Tech Stack:** Python 3.12+, SymPy 1.14+, MLIR/C++, pytest

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `python/lego/c_printer.py` | B1 fix + A1 refactor |
| Modify | `python/lego/cxx_printer.py` | B1 fix + A1 refactor |
| Modify | `python/lego/rust_printer.py` | B1 fix + A1 refactor |
| Modify | `python/lego/js_printer.py` | B1 fix + A1 refactor |
| Modify | `python/lego/fortran_printer.py` | B1 fix + A1 refactor |
| Modify | `python/lego/glsl_printer.py` | B1 fix + A1 refactor |
| Modify | `python/lego/julia_printer.py` | B1 fix + A1 refactor |
| Modify | `python/lego/python_printer.py` | B1 fix |
| Modify | `python/lego/backend/symbolic.py` | B2 fix + A4 wildcard import |
| Modify | `python/lego/core.py` | B3 dead code removal |
| Modify | `include/Lego/Passes.h` | B4 ghost pass removal |
| Modify | `python/examples/symbolic/graphene.py` | B6 typo fix |
| Modify | `python/examples/jax/hello_world.py` | B7 spurious import |
| Create | `python/lego/_printer_base.py` | A1 shared printer mixin |
| Modify | `python/lego/frontends/_adapter.py` | A2 shared temp-file helper |
| Modify | `python/lego/frontends/triton_jit.py` | A2 + A3 refactor |
| Modify | `python/lego/frontends/cutile_jit.py` | A2 refactor |
| Modify | `python/lego/rewriter.py` | A3 remove Triton import |
| Create | `python/tests/test_printer_kwargs.py` | B1 regression test |
| Modify | `python/tests/test_codegen_backends.py` | A1 regression test |

---

### Task 1: Fix `super().__init__` kwargs bug in all printers (B1)

**Files:**
- Modify: `python/lego/python_printer.py:8`
- Modify: `python/lego/c_printer.py:7`
- Modify: `python/lego/cxx_printer.py:9`
- Modify: `python/lego/rust_printer.py:9`
- Modify: `python/lego/js_printer.py:9`
- Modify: `python/lego/fortran_printer.py:11`
- Modify: `python/lego/glsl_printer.py:9`
- Modify: `python/lego/julia_printer.py:9`
- Test: `python/tests/test_printer_kwargs.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_printer_kwargs.py`:

```python
"""Regression test for B1: printer kwargs must be forwarded correctly."""
import sympy as sp
from lego.c_printer import LEGOCCodePrinter
from lego.cxx_printer import LEGOCXXCodePrinter
from lego.rust_printer import LEGORustCodePrinter
from lego.js_printer import LEGOJSCodePrinter
from lego.fortran_printer import LEGOFortranCodePrinter
from lego.glsl_printer import LEGOGLSLCodePrinter
from lego.julia_printer import LEGOJuliaCodePrinter
from lego.python_printer import LEGOPythonCodePrinter
import pytest


@pytest.mark.parametrize("PrinterClass", [
    LEGOCCodePrinter,
    LEGOCXXCodePrinter,
    LEGORustCodePrinter,
    LEGOJSCodePrinter,
    LEGOFortranCodePrinter,
    LEGOGLSLCodePrinter,
    LEGOJuliaCodePrinter,
    LEGOPythonCodePrinter,
])
def test_printer_accepts_kwargs(PrinterClass):
    """All printers must forward kwargs to their SymPy parent without error."""
    # The settings kwarg is accepted by all SymPy printers.
    # Before the fix, kwargs was passed as a positional arg, causing
    # silent corruption or TypeError.
    printer = PrinterClass(settings={"full_prec": True})
    x = sp.Symbol("x")
    result = printer.doprint(x)
    assert "x" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && python -m pytest python/tests/test_printer_kwargs.py -v`

Expected: FAIL with TypeError from SymPy receiving a dict as a positional arg.

- [ ] **Step 3: Fix all 8 printer files**

In each file, change `super().__init__(*args, kwargs)` to `super().__init__(*args, **kwargs)`:

- `python/lego/python_printer.py:8`
- `python/lego/c_printer.py:7`
- `python/lego/cxx_printer.py:9`
- `python/lego/rust_printer.py:9`
- `python/lego/js_printer.py:9`
- `python/lego/fortran_printer.py:11`
- `python/lego/glsl_printer.py:9`
- `python/lego/julia_printer.py:9`

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && python -m pytest python/tests/test_printer_kwargs.py -v`

Expected: PASS for all 8 printers.

- [ ] **Step 5: Run existing tests for regressions**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && python -m pytest python/tests/test_codegen_backends.py python/tests/test_rewriter.py -v`

Expected: All existing tests still pass.

- [ ] **Step 6: Commit**

```
git add python/lego/python_printer.py python/lego/c_printer.py python/lego/cxx_printer.py python/lego/rust_printer.py python/lego/js_printer.py python/lego/fortran_printer.py python/lego/glsl_printer.py python/lego/julia_printer.py python/tests/test_printer_kwargs.py
git commit -m "fix: forward **kwargs in all printer __init__ methods (B1)"
```

---

### Task 2: Fix unsigned division in symbolic.py (B2)

**Files:**
- Modify: `python/lego/backend/symbolic.py:119,133,137`
- Test: `python/tests/test_mlir_roundtrip.py`

- [ ] **Step 1: Write the regression test**

Add to `python/tests/test_mlir_roundtrip.py`:

```python
def test_signed_division_roundtrip():
    """B2: verify MLIR roundtrip uses signed division for index arithmetic."""
    from lego.core import Row
    from lego.backend.symbolic import simplify_via_mlir
    import sympy as sp

    M, N = sp.symbols('M N', positive=True, integer=True)
    flat = sp.Symbol('flat', nonneg=True, integer=True)
    result = simplify_via_mlir(Row(M, N), 'inv', flat,
                                constraints={flat: (0, M * N)})
    assert len(result) == 2
    subs = {M: 4, N: 3, flat: 7}
    assert int(result[0].subs(subs)) == 2  # 7 // 3 = 2
    assert int(result[1].subs(subs)) == 1  # 7 % 3 = 1
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && python -m pytest python/tests/test_mlir_roundtrip.py::test_signed_division_roundtrip -v`

Expected: May pass for non-negative inputs. The fix is for correctness with negative offsets.

- [ ] **Step 3: Change `divui` to `divsi` and `remui` to `remsi`**

In `python/lego/backend/symbolic.py`:

Line 119: `arith.divui(` -> `arith.divsi(`
Line 133: `arith.divui(` -> `arith.divsi(`
Line 137: `arith.remui(` -> `arith.remsi(`

- [ ] **Step 4: Run full roundtrip tests**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && python -m pytest python/tests/test_mlir_roundtrip.py -v`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```
git add python/lego/backend/symbolic.py python/tests/test_mlir_roundtrip.py
git commit -m "fix: use signed division (divsi/remsi) in symbolic MLIR lowering (B2)"
```

---

### Task 3: Remove dead code in GroupBy (B3)

**Files:**
- Modify: `python/lego/core.py:448-461`

- [ ] **Step 1: Verify the methods are dead code**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && grep -rn 'GroupBy.*\.transform\|GroupBy.*\.inverse_transform' python/lego/ python/tests/ --include='*.py' | grep -v 'core.py'`

Expected: No results confirming nothing calls these methods.

- [ ] **Step 2: Delete the dead methods**

In `python/lego/core.py`, remove lines 448-461:

```python
# DELETE these methods:
    def transform(self, tensor):
        """Apply layout transform via MLIR JIT compilation."""
        from .backend.compiler import get_compiler
        if not hasattr(self, '_compiled'):
            self._compiled = get_compiler(self, tensor.shape)
        return self._compiled.transform_numpy(tensor) if hasattr(tensor, 'ctypes') \
            else self._compiled.transform_numpy(tensor)

    def inverse_transform(self, tensor):
        """Apply inverse layout transform."""
        from .backend.compiler import get_compiler
        if not hasattr(self, '_compiled'):
            self._compiled = get_compiler(self, tensor.shape)
        return self._compiled.inverse_transform_numpy(tensor)
```

- [ ] **Step 3: Run existing tests**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && python -m pytest python/tests/ -v --timeout=60`

Expected: All tests pass.

- [ ] **Step 4: Commit**

```
git add python/lego/core.py
git commit -m "fix: remove dead GroupBy.transform/inverse_transform methods (B3)"
```

---

### Task 4: Remove ghost pass declaration (B4)

**Files:**
- Modify: `include/Lego/Passes.h:19`

- [ ] **Step 1: Verify the pass is never defined**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && grep -rn 'VerifyGenpConsistency' lib/ include/ tools/`

Expected: Only the declaration in `Passes.h:19`.

- [ ] **Step 2: Remove the declaration**

In `include/Lego/Passes.h`, delete line 19:

```cpp
std::unique_ptr<Pass> createLegoVerifyGenpConsistencyPass();
```

- [ ] **Step 3: Build to verify**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && cmake --build build --target lego-opt 2>&1 | tail -5`

Expected: Build succeeds.

- [ ] **Step 4: Commit**

```
git add include/Lego/Passes.h
git commit -m "fix: remove undeclared createLegoVerifyGenpConsistencyPass (B4)"
```

---

### Task 5: Fix typo and spurious import (B6, B7)

**Files:**
- Modify: `python/examples/symbolic/graphene.py:3`
- Modify: `python/examples/jax/hello_world.py`

- [ ] **Step 1: Fix the typo in graphene.py**

In `python/examples/symbolic/graphene.py`, line 3:

```python
# Before:
i, j, k, w, q, n = symbols('i j k w q n ', integer=True, postive=True)
# After:
i, j, k, w, q, n = symbols('i j k w q n ', integer=True, positive=True)
```

- [ ] **Step 2: Remove torch from JAX example**

In `python/examples/jax/hello_world.py`:

Delete line 5 (`import torch`).

Replace the `__main__` block to use numpy instead of torch:

```python
if __name__ == "__main__":
    np.random.seed(0)
    for N in [1024, 8192, 2**16]:
        x_np = np.random.randn(N).astype(np.float32)
        y_np = np.random.randn(N).astype(np.float32)
        expected = x_np + y_np

        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        z_jax = vecadd(x_jax, y_jax, N)
        z_np = np.asarray(z_jax)

        ok = np.allclose(z_np, expected, atol=1e-5)
        print(f"N={N:>6d}  match={ok}")
        assert ok, f"Mismatch at N={N}"
    print("PASS: JAX vecadd matches NumPy")
```

- [ ] **Step 3: Commit**

```
git add python/examples/symbolic/graphene.py python/examples/jax/hello_world.py
git commit -m "fix: typo in graphene.py and remove torch from JAX example (B6, B7)"
```

---

### Task 6: Extract `LEGOStaticLangMixin` base class (A1)

**Files:**
- Create: `python/lego/_printer_base.py`
- Modify: `python/lego/c_printer.py`
- Modify: `python/lego/cxx_printer.py`
- Modify: `python/lego/rust_printer.py`
- Modify: `python/lego/js_printer.py`
- Modify: `python/lego/fortran_printer.py`
- Modify: `python/lego/glsl_printer.py`
- Modify: `python/lego/julia_printer.py`
- Test: `python/tests/test_codegen_backends.py` (existing)

- [ ] **Step 1: Create the shared mixin**

Create `python/lego/_printer_base.py`:

```python
"""Shared mixin for static-language LEGO code printers.

Provides common implementations of _print_BroadcastRange, _print_floor,
and _print_Mod that most static languages share. Language-specific
printers override only the parts that differ.
"""


class LEGOStaticLangMixin:
    """Mixin providing common print methods for static-language printers.

    Mix this in BEFORE the SymPy printer base class so these methods take
    precedence via MRO. Each method can be overridden in the concrete printer.
    """

    def _print_BroadcastRange(self, expr):
        return f"({self._print(expr.args[0])})"

    def _print_floor(self, expr):
        arg = expr.args[0]
        num, den = arg.as_numer_denom()
        return f"(({self._print(num)}) / ({self._print(den)}))"

    def _print_Mod(self, expr):
        return f"(({self._print(expr.args[0])}) % ({self._print(expr.args[1])}))"
```

- [ ] **Step 2: Refactor `cxx_printer.py`**

Replace full file with:

```python
"""C++ code printer for LEGO layout expressions."""

from sympy.printing.cxx import CXX17CodePrinter
from ._printer_base import LEGOStaticLangMixin


class LEGOCXXCodePrinter(LEGOStaticLangMixin, CXX17CodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_Pow(self, expr):
        from sympy.printing.precedence import precedence
        from sympy.core.numbers import equal_valued

        PREC = precedence(expr)

        if equal_valued(expr.exp, 2):
            base = self.parenthesize(expr.base, PREC)
            return f"{base} * {base}"
        elif equal_valued(expr.exp, -1):
            return f"1.0 / {self.parenthesize(expr.base, PREC)}"
        elif equal_valued(expr.exp, 0.5):
            return f"std::sqrt(static_cast<double>({self._print(expr.base)}))"
        elif equal_valued(expr.exp, 1.0 / 3):
            return f"std::cbrt(static_cast<double>({self._print(expr.base)}))"
        else:
            return f"std::pow(static_cast<double>({self._print(expr.base)}), {self._print(expr.exp)})"

    def _print_lego_arange(self, expr):
        return f"std::views::iota({self._print(expr.args[0])}, {self._print(expr.args[1])})"
```

- [ ] **Step 3: Refactor `rust_printer.py`**

Replace full file with:

```python
"""Rust code printer for LEGO layout expressions."""

from sympy.printing.rust import RustCodePrinter
from ._printer_base import LEGOStaticLangMixin


class LEGORustCodePrinter(LEGOStaticLangMixin, RustCodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_Pow(self, expr):
        from sympy.core.numbers import equal_valued

        if equal_valued(expr.exp, 2):
            base = self._print(expr.base)
            return f"({base} * {base})"
        elif equal_valued(expr.exp, -1):
            return f"(1.0 / {self._print(expr.base)})"
        elif equal_valued(expr.exp, 0.5):
            return f"({self._print(expr.base)} as f64).sqrt()"
        else:
            return f"({self._print(expr.base)} as f64).powi({self._print(expr.exp)})"

    def _print_lego_arange(self, expr):
        return f"({self._print(expr.args[0])}..{self._print(expr.args[1])})"
```

- [ ] **Step 4: Refactor `js_printer.py`**

Replace full file with:

```python
"""JavaScript code printer for LEGO layout expressions."""

from sympy.printing.jscode import JavascriptCodePrinter
from ._printer_base import LEGOStaticLangMixin


class LEGOJSCodePrinter(LEGOStaticLangMixin, JavascriptCodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_floor(self, expr):
        arg = expr.args[0]
        num, den = arg.as_numer_denom()
        return f"Math.floor(({self._print(num)}) / ({self._print(den)}))"

    def _print_lego_arange(self, expr):
        start = self._print(expr.args[0])
        stop = self._print(expr.args[1])
        return f"Array.from({{length: {stop} - {start}}}, (_, i) => i + {start})"
```

- [ ] **Step 5: Refactor `glsl_printer.py`**

Replace full file with:

```python
"""GLSL code printer for LEGO layout expressions."""

from sympy.printing.glsl import GLSLPrinter
from ._printer_base import LEGOStaticLangMixin


class LEGOGLSLCodePrinter(LEGOStaticLangMixin, GLSLPrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_lego_arange(self, expr):
        return f"/* arange({self._print(expr.args[0])}, {self._print(expr.args[1])}) */"
```

- [ ] **Step 6: Refactor `julia_printer.py`**

Replace full file with:

```python
"""Julia code printer for LEGO layout expressions."""

from sympy.printing.julia import JuliaCodePrinter
from ._printer_base import LEGOStaticLangMixin


class LEGOJuliaCodePrinter(LEGOStaticLangMixin, JuliaCodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_floor(self, expr):
        arg = expr.args[0]
        num, den = arg.as_numer_denom()
        return f"div({self._print(num)}, {self._print(den)})"

    def _print_Mod(self, expr):
        return f"mod({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_lego_arange(self, expr):
        start = self._print(expr.args[0])
        stop = expr.args[1]
        stop_str = self._print(stop - 1) if hasattr(stop, '__sub__') else f"({self._print(stop)} - 1)"
        return f"({start}:{stop_str})"
```

- [ ] **Step 7: Refactor `fortran_printer.py`**

Replace full file with:

```python
"""Fortran code printer for LEGO layout expressions."""

from sympy.printing.fortran import FCodePrinter
from ._printer_base import LEGOStaticLangMixin


class LEGOFortranCodePrinter(LEGOStaticLangMixin, FCodePrinter):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('standard', 95)
        super().__init__(*args, **kwargs)

    def _print_Pow(self, expr):
        from sympy.core.numbers import equal_valued

        if equal_valued(expr.exp, 2):
            base = self._print(expr.base)
            return f"({base}**2)"
        elif equal_valued(expr.exp, 0.5):
            return f"sqrt(dble({self._print(expr.base)}))"
        else:
            return f"(({self._print(expr.base)})**({self._print(expr.exp)}))"

    def _print_Mod(self, expr):
        return f"mod({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_lego_arange(self, expr):
        start = self._print(expr.args[0])
        stop = expr.args[1]
        stop_str = self._print(stop - 1) if hasattr(stop, '__sub__') else f"({self._print(stop)} - 1)"
        return f"(/ (i, i = {start}, {stop_str}) /)"
```

- [ ] **Step 8: Refactor `c_printer.py`**

Replace full file with:

```python
"""C99 code printer for LEGO layout expressions."""

from sympy.printing.c import C99CodePrinter
from ._printer_base import LEGOStaticLangMixin


class LEGOCCodePrinter(LEGOStaticLangMixin, C99CodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_Pow(self, expr):
        from sympy.printing.precedence import precedence
        from sympy.core.numbers import equal_valued
        from sympy.codegen.ast import real

        if "Pow" in self.known_functions:
            return self._print_Function(expr)

        PREC = precedence(expr)
        suffix = self._get_func_suffix(real)

        if equal_valued(expr.exp, 2):
            return '%s*%s' % (self.parenthesize(expr.base, PREC), self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, -1):
            literal_suffix = self._get_literal_suffix(real)
            return '1.0%s/%s' % (literal_suffix, self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return '%ssqrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        elif expr.exp == 1/3 and self.standard != 'C89':
            return '%scbrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        else:
            return '%spow%s(%s, %s)' % (self._ns, suffix, self._print(expr.base),
                                        self._print(expr.exp))
```

- [ ] **Step 9: Run all tests**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && python -m pytest python/tests/test_codegen_backends.py python/tests/test_printer_kwargs.py python/tests/test_rewriter.py -v`

Expected: All tests pass.

- [ ] **Step 10: Commit**

```
git add python/lego/_printer_base.py python/lego/c_printer.py python/lego/cxx_printer.py python/lego/rust_printer.py python/lego/js_printer.py python/lego/fortran_printer.py python/lego/glsl_printer.py python/lego/julia_printer.py
git commit -m "refactor: extract LEGOStaticLangMixin to eliminate printer duplication (A1)"
```

---

### Task 7: Extract shared temp-file helper (A2)

**Files:**
- Modify: `python/lego/frontends/_adapter.py`
- Modify: `python/lego/frontends/triton_jit.py:254-299`
- Modify: `python/lego/frontends/cutile_jit.py:91-128`

- [ ] **Step 1: Add the shared helper to `_adapter.py`**

Add at the end of `python/lego/frontends/_adapter.py`:

```python
def write_and_exec_temp_file(new_source, tree, original_fn, return_source=False):
    """Write transformed source to a temp file, compile, and exec it.

    Shared by Triton and cuTile adapters. Returns (source_text, None)
    if return_source is True, otherwise (namespace, transformed_fn).
    """
    import os
    import atexit

    _save = os.environ.get('LEGO_SAVE_KERNEL', False)
    temp_dir = os.environ.get("LEGO_TEMP_DIR", "/tmp/lego_kernels")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(
        temp_dir, f"{original_fn.__name__}_{id(original_fn)}.py")

    with open(temp_file, 'w') as f:
        f.write(new_source)

    if return_source:
        if not _save:
            os.remove(temp_file)
        return new_source, None

    code_obj = compile(tree, filename=temp_file, mode='exec')
    namespace = original_fn.__globals__.copy()
    exec(code_obj, namespace)  # noqa: S102

    if not _save:
        atexit.register(
            lambda f=temp_file: os.remove(f) if os.path.exists(f) else None)

    transformed_fn = namespace[original_fn.__name__]
    transformed_fn.__code__ = transformed_fn.__code__.replace(
        co_filename=temp_file)

    return namespace, transformed_fn
```

- [ ] **Step 2: Refactor `triton_jit.py` compile_and_wrap**

Replace the body of `compile_and_wrap` in `python/lego/frontends/triton_jit.py` (lines 254-299):

```python
    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        from lego.frontends._adapter import write_and_exec_temp_file

        result, transformed_fn = write_and_exec_temp_file(
            new_source, tree, original_fn, return_source)
        if return_source:
            return result

        # Re-apply Triton wrappers in reverse order
        if wrappers:
            import triton
            from triton.runtime.jit import JITFunction
            from triton.runtime.autotuner import Autotuner
            for wrapper in reversed(wrappers):
                if isinstance(wrapper, Autotuner):
                    transformed_fn = triton.autotune(
                        configs=wrapper.configs,
                        key=wrapper.keys,
                    )(transformed_fn)
                elif isinstance(wrapper, JITFunction):
                    transformed_fn = triton.jit(transformed_fn)

        return transformed_fn
```

- [ ] **Step 3: Refactor `cutile_jit.py` compile_and_wrap**

Replace the body of `compile_and_wrap` in `python/lego/frontends/cutile_jit.py` (lines 91-128):

```python
    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        from lego.frontends._adapter import write_and_exec_temp_file

        result, transformed_fn = write_and_exec_temp_file(
            new_source, tree, original_fn, return_source)
        if return_source:
            return result

        # Re-apply cuTile wrappers in reverse order
        if wrappers:
            import cuda.tile as ct_mod
            for wrapper in reversed(wrappers):
                transformed_fn = ct_mod.kernel(transformed_fn)

        return transformed_fn
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && python -m pytest python/tests/test_rewriter.py python/tests/test_codegen_backends.py -v`

Expected: All pass.

- [ ] **Step 5: Commit**

```
git add python/lego/frontends/_adapter.py python/lego/frontends/triton_jit.py python/lego/frontends/cutile_jit.py
git commit -m "refactor: extract shared temp-file helper from Triton/cuTile adapters (A2)"
```

---

### Task 8: Fix Triton import in shared rewriter (A3)

**Files:**
- Modify: `python/lego/rewriter.py:128`
- Modify: `python/lego/frontends/_adapter.py`
- Modify: `python/lego/frontends/triton_jit.py`

- [ ] **Step 1: Add `try_block_ptr_pattern` hook to `DSLAdapter`**

In `python/lego/frontends/_adapter.py`, add to the `DSLAdapter` class after `get_rewriter_options`:

```python
    def try_block_ptr_pattern(self, stmt, eval_env, param_names, block_ptr_targets):
        """Try to match a block-ptr pattern in the AST.

        Only the Triton adapter overrides this. Other DSLs return None.
        """
        return None
```

- [ ] **Step 2: Override in `TritonAdapter`**

In `python/lego/frontends/triton_jit.py`, add to the `TritonAdapter` class:

```python
    def try_block_ptr_pattern(self, stmt, eval_env, param_names, block_ptr_targets):
        from lego.frontends.triton_jit import extract_block_ptr_metadata
        return _try_block_ptr_pattern_impl(stmt, eval_env, param_names, block_ptr_targets)
```

Where `_try_block_ptr_pattern_impl` contains the logic currently at `rewriter.py:123+`.

- [ ] **Step 3: Update the rewriter to use the adapter hook**

In `python/lego/rewriter.py`, change the `_try_block_ptr_pattern` function to accept an `adapter` parameter and delegate to `adapter.try_block_ptr_pattern(...)` instead of importing from `triton_jit` directly. Remove the direct import on line 128.

- [ ] **Step 4: Run tests**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && python -m pytest python/tests/test_rewriter.py python/tests/test_block_ptr.py -v`

Expected: All pass.

- [ ] **Step 5: Commit**

```
git add python/lego/rewriter.py python/lego/frontends/_adapter.py python/lego/frontends/triton_jit.py
git commit -m "refactor: move block-ptr pattern matching from rewriter into TritonAdapter (A3)"
```

---

### Task 9: Replace wildcard import in symbolic.py (A4)

**Files:**
- Modify: `python/lego/backend/symbolic.py:4`

- [ ] **Step 1: Find which names are used**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && grep -oP '(?<!\w)(Row|Col|RegP|GenP|OrderBy|GroupBy|TileByLayout|BroadcastRange|lego_arange|symbols)(?!\w)' python/lego/backend/symbolic.py | sort -u`

- [ ] **Step 2: Replace the wildcard import**

In `python/lego/backend/symbolic.py`, change line 4:

```python
# Before:
from lego.core import *
# After (adjust based on Step 1 results):
from lego.core import (
    Row, Col, RegP, GenP, OrderBy, GroupBy, TileByLayout,
    BroadcastRange, lego_arange,
)
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && python -m pytest python/tests/test_mlir_roundtrip.py python/tests/test_rewriter.py -v`

Expected: All pass.

- [ ] **Step 4: Commit**

```
git add python/lego/backend/symbolic.py
git commit -m "refactor: replace wildcard import in symbolic.py with explicit imports (A4)"
```

---

### Task 10: Restore bench_utils.py for puzzles (B5)

**Files:**
- Location: `python/examples/puzzles/`

- [ ] **Step 1: Check git history for the source file**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && git log --all --diff-filter=D -- '**/bench_utils.py' | head -10`

If not found, try decompiling the .pyc:

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO && python -c "import dis, marshal; f=open('python/examples/puzzles/__pycache__/bench_utils.cpython-314.pyc','rb'); f.read(16); code=marshal.load(f); dis.dis(code)" 2>&1 | head -40`

- [ ] **Step 2: Restore the source file**

If found in git history:
```
git show <commit>:python/examples/puzzles/bench_utils.py > python/examples/puzzles/bench_utils.py
```

If not, reconstruct from the .pyc bytecode or from the usage pattern (all puzzles do `from bench_utils import run_benchmark`).

- [ ] **Step 3: Verify import works**

Run: `cd /Users/amirmohammadtavakkoli/project/LEGO/python/examples/puzzles && python -c "from bench_utils import run_benchmark; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```
git add python/examples/puzzles/bench_utils.py
git commit -m "fix: restore bench_utils.py source for puzzle examples (B5)"
```
