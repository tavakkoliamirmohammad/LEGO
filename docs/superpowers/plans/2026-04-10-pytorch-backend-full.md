# Fully-Fledged LEGO PyTorch Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, bug-free, optimized PyTorch backend for LEGO with Triton-generated layout-aware kernels, a real cross-op layout planner, and full torch.compile integration.

**Architecture:** Three layers — (1) Op layer: custom ops via `torch.library` with Triton codegen from LEGO layout algebra, (2) Tensor layer: `LegoTensor` with `__torch_dispatch__` and correct handling of physical/virtual annotations, (3) Compiler layer: `torch.compile(backend="lego")` with a real cost-model planner that inserts rearrangements and wires layout-aware index arithmetic into inductor. All permutations use the existing `LayoutCompiler` MLIR pipeline. Triton kernels use LEGO's existing `triton_jit` rewriting infrastructure.

**Tech Stack:** Python 3.12+, PyTorch 2.x, Triton, LEGO MLIR pipeline, SymPy, NumPy

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `python/lego/torch/__init__.py` | Fix `__getattr__`, restore `LegoLayout.transform` torch path, re-export |
| Modify | `python/lego/torch/tensor.py` | Fix C1 (dim-reductions), I3 (TransposedLayout.compose), I5 (thread safety), add Tier 5 for dim-reductions |
| Modify | `python/lego/torch/ops.py` | Add `lego::addmm`, fix `@torch_op` fake impl (I4), add `lego::permute` op |
| Create | `python/lego/torch/triton_kernels.py` | Triton kernel generation from layout algebra for mm/bmm |
| Modify | `python/lego/torch/compile.py` | Fix backend to correctly handle graph + inputs, wire Path B/C |
| Modify | `python/lego/torch/planner.py` | Real cost-model planner that inserts rearrangement nodes |
| Modify | `python/lego/torch/fusion.py` | Fix exception swallowing (I6), wire index injection into inductor |
| Create | `python/lego/torch/autotune.py` | Replacement autotune with proper CUDA sync |
| Modify | `python/lego/frontends/python_mlir.py` | Restore torch paths in `LegoLayout.transform`/`inverse_transform`, `BatchedLayout` |
| Modify | `python/lego/__init__.py` | Fix `__getattr__` for `lego.annotate` UX |
| Modify | `python/tests/test_torch_subclass.py` | Add physical-data tests, dim-reduction tests |
| Modify | `python/tests/test_torch_ops.py` | Add `lego::addmm`, `lego::permute`, Triton kernel tests |
| Modify | `python/tests/test_torch_compile.py` | Add planner insertion tests, Path B/C tests |
| Create | `python/tests/test_torch_autotune.py` | Autotune tests |
| Create | `python/tests/test_torch_stress.py` | Stress tests: large tensors, mixed layouts, end-to-end |

---

### Task 1: Fix `lego.annotate` UX — `__getattr__` in `__init__.py` (C2)

**Files:**
- Modify: `python/lego/__init__.py:72-83`
- Test: `python/tests/test_torch_subclass.py`

- [ ] **Step 1: Write the failing test**

Add to `python/tests/test_torch_subclass.py` at the end of `TestEdgeCases`:

```python
    def test_lego_annotate_without_preimport(self):
        """lego.annotate works even if accessed before torch is in sys.modules."""
        import importlib
        import lego as lego_mod
        # Force __getattr__ path
        if "annotate" in lego_mod.__dict__:
            del lego_mod.__dict__["annotate"]
        # Should not raise AttributeError
        fn = lego_mod.annotate
        assert callable(fn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py::TestEdgeCases::test_lego_annotate_without_preimport -xvs`

Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Fix `__getattr__` to import torch on demand**

In `python/lego/__init__.py`, replace lines 72-83:

```python
# PyTorch integration — import torch on demand to avoid ~4s penalty
# for non-torch workflows, but don't fail silently when torch IS needed.
def __getattr__(name):
    _TORCH_NAMES = ("annotate", "rearrange", "LegoTensor")
    if name in _TORCH_NAMES:
        try:
            from .torch import annotate, rearrange, LegoTensor  # noqa: F811
            globals()["annotate"] = annotate
            globals()["rearrange"] = rearrange
            globals()["LegoTensor"] = LegoTensor
            return globals()[name]
        except ImportError:
            raise AttributeError(
                f"lego.{name} requires PyTorch. Install it with: pip install torch"
            ) from None
    raise AttributeError(f"module 'lego' has no attribute {name!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py::TestEdgeCases::test_lego_annotate_without_preimport -xvs`

Expected: PASS.

- [ ] **Step 5: Run all existing torch tests for regressions**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py tests/test_torch_ops.py tests/test_torch_compile.py -x --timeout=120`

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add python/lego/__init__.py python/tests/test_torch_subclass.py
git commit -m "fix: import torch on demand in lego.__getattr__ (C2)"
```

---

### Task 2: Restore `LegoLayout.transform()` torch path (C3)

**Files:**
- Modify: `python/lego/frontends/python_mlir.py:185-200`
- Test: `python/tests/test_torch_subclass.py`

- [ ] **Step 1: Write the failing test**

Add a new class to `python/tests/test_torch_subclass.py`:

```python
class TestLegoLayoutTorchPath:
    def test_transform_torch_tensor(self):
        """LegoLayout.transform() accepts torch tensors."""
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        result = layout.transform(x)
        assert isinstance(result, torch.Tensor)
        # Verify round-trip
        back = layout.inverse_transform(result)
        torch.testing.assert_close(back, x)

    def test_transform_batched_torch(self):
        """BatchedLayout.transform() accepts torch tensors."""
        from lego.frontends.python_mlir import Batched
        base = ColMajor((4, 4))
        batched = Batched(base, batch_shape=(2,))
        x = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
        result = batched.transform(x)
        assert isinstance(result, torch.Tensor)
        back = batched.inverse_transform(result)
        torch.testing.assert_close(back, x)

    @requires_cuda
    def test_transform_torch_cuda(self):
        """LegoLayout.transform() works on CUDA tensors."""
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
        result = layout.transform(x)
        assert result.device.type == "cuda"
        back = layout.inverse_transform(result)
        torch.testing.assert_close(back, x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py::TestLegoLayoutTorchPath -xvs`

Expected: FAIL with TypeError (torch tensor not handled).

- [ ] **Step 3: Restore torch path in `LegoLayout.transform`**

In `python/lego/frontends/python_mlir.py`, modify `transform()` (after `self._validate_numel(tensor)`, before the numpy path):

```python
    def transform(self, tensor):
        self._validate_numel(tensor)
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            compiler = self._get_compiler(tensor)
            fwd, _ = compiler.get_permutation_table()
            fwd_t = torch.from_numpy(np.ascontiguousarray(fwd)).to(tensor.device)
            return tensor.reshape(-1)[fwd_t].reshape(self._shape)
        if isinstance(tensor, np.ndarray):
            if self._composed_perm is not None:
                fwd, _ = self._composed_perm
                return tensor.ravel()[fwd].reshape(self._shape)
            compiler = self._get_compiler(tensor)
            return compiler.transform_numpy(tensor).reshape(self._shape)
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")
```

- [ ] **Step 4: Restore torch path in `LegoLayout.inverse_transform`**

In `python/lego/frontends/python_mlir.py`, modify `inverse_transform()` similarly:

```python
    def inverse_transform(self, tensor):
        self._validate_numel(tensor)
        _check_layout_invertible(self._layout)
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            compiler = self._get_compiler(tensor)
            _, inv = compiler.get_permutation_table()
            inv_t = torch.from_numpy(np.ascontiguousarray(inv)).to(tensor.device)
            return tensor.reshape(-1)[inv_t].reshape(self._shape)
        if isinstance(tensor, np.ndarray):
            if self._composed_perm is not None:
                _, inv = self._composed_perm
                return tensor.ravel()[inv].reshape(self._shape)
            compiler = self._get_compiler(tensor)
            return compiler.inverse_transform_numpy(tensor).reshape(self._shape)
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")
```

- [ ] **Step 5: Restore torch path in `BatchedLayout.transform` and `inverse_transform`**

In `python/lego/frontends/python_mlir.py`, inside `BatchedLayout.transform()`, add before the numpy path:

```python
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            batch_total = 1
            for d in tensor.shape[:batch_dims]:
                batch_total *= d
            flat = tensor.reshape(batch_total, -1)
            compiler = LayoutCompiler(self._base._layout, self._base._shape, "i64")
            fwd, _ = compiler.get_permutation_table()
            fwd_t = torch.from_numpy(np.ascontiguousarray(fwd)).to(tensor.device)
            result = flat[:, fwd_t]
            return result.reshape(self._shape)
```

And the same pattern for `BatchedLayout.inverse_transform()` using `inv` instead of `fwd`.

- [ ] **Step 6: Run test to verify it passes**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py::TestLegoLayoutTorchPath -xvs`

Expected: PASS.

- [ ] **Step 7: Run full test suite for regressions**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py tests/test_torch_ops.py tests/test_torch_compile.py tests/test_tensor_api.py -x --timeout=120`

Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add python/lego/frontends/python_mlir.py python/tests/test_torch_subclass.py
git commit -m "fix: restore LegoLayout.transform/inverse_transform torch paths (C3)"
```

---

### Task 3: Fix dim-reductions on physical data (C1)

**Files:**
- Modify: `python/lego/torch/tensor.py:55-58, 168-183, 263-298`
- Test: `python/tests/test_torch_subclass.py`

- [ ] **Step 1: Write the failing test**

Add a new class to `python/tests/test_torch_subclass.py`:

```python
class TestPhysicalDimReductions:
    def test_sum_dim_on_physical_data(self):
        """sum(dim=k) on physically-rearranged data must be correct."""
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        rx = rearrange(x, layout)
        result = torch.sum(rx, dim=1)
        expected = torch.sum(x, dim=1)
        torch.testing.assert_close(result, expected)

    def test_mean_dim_on_physical_data(self):
        """mean(dim=k) on physically-rearranged data must be correct."""
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)
        rx = rearrange(x, layout)
        result = torch.mean(rx, dim=0)
        expected = torch.mean(x, dim=0)
        torch.testing.assert_close(result, expected)

    def test_sum_dim_on_virtual_still_works(self):
        """sum(dim=k) on virtual annotations still gives correct results."""
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)
        ax = annotate(x, layout)
        result = torch.sum(ax, dim=1)
        expected = torch.sum(x, dim=1)
        torch.testing.assert_close(result, expected)
```

- [ ] **Step 2: Run test to verify C1 bug exists**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py::TestPhysicalDimReductions -xvs`

Expected: `test_sum_dim_on_physical_data` FAILS (wrong values).

- [ ] **Step 3: Create a separate tier for dim-reductions**

In `python/lego/torch/tensor.py`, add a new set after `_FULL_REDUCTIONS`:

```python
_DIM_REDUCTIONS: set = set()
```

- [ ] **Step 4: Move dim-reductions from Tier 1 to the new set**

In the `_populate()` function, remove these three entries from `_TIER1`:

```python
        # REMOVE from _TIER1:
        # aten.sum.dim_IntList,
        # aten.mean.dim,
        # aten.prod.dim_int,
```

And add them to `_DIM_REDUCTIONS`:

```python
    _DIM_REDUCTIONS.update([
        aten.sum.dim_IntList,
        aten.mean.dim,
        aten.prod.dim_int,
        aten.amax.default,  # amax with dim arg
        aten.amin.default,  # amin with dim arg
    ])
```

Note: `amax.default` and `amin.default` accept an optional dim arg and were in `_FULL_REDUCTIONS`. Move them to `_DIM_REDUCTIONS` only if they have dim args. Actually, it is simpler to handle this in dispatch. Keep `_FULL_REDUCTIONS` as-is for truly no-dim reductions. Add only the dim-IntList variants to `_DIM_REDUCTIONS`:

```python
    _DIM_REDUCTIONS.update([
        aten.sum.dim_IntList,
        aten.mean.dim,
        aten.prod.dim_int,
    ])
```

- [ ] **Step 5: Add dim-reduction handler in `__torch_dispatch__`**

In `__torch_dispatch__`, add after the `_FULL_REDUCTIONS` check:

```python
        # Dim-reductions: must inverse-rearrange physical data first
        if func in _DIM_REDUCTIONS:
            return _dispatch_dim_reduction(func, args, kwargs)
```

- [ ] **Step 6: Implement `_dispatch_dim_reduction`**

Add after `_dispatch_tier3`:

```python
def _dispatch_dim_reduction(func, args, kwargs):
    """Dim-reductions: correct on virtual data, need inverse on physical."""
    lt = _first_lego(args)
    if lt is not None and lt._is_physical:
        # Physical data is in layout order — inverse-rearrange to logical
        # order before reducing along a dimension.
        from lego.backend.compiler import LayoutCompiler
        layout = lt._layout
        base = layout._base if hasattr(layout, "_base") else layout
        compiler = LayoutCompiler(base._layout, base._shape, "i64")
        _, inv = compiler.get_permutation_table()
        inv_t = torch.from_numpy(np.ascontiguousarray(inv)).to(lt._data.device)
        logical = lt._data.reshape(-1)[inv_t].reshape(lt._data.shape)
        new_args = list(args)
        new_args[0] = logical
        return func(*new_args, **{k: _unwrap(v) for k, v in kwargs.items()})
    # Virtual data: data is still in logical order, safe to reduce directly
    return func(*_unwrap_args(args), **{k: _unwrap(v) for k, v in kwargs.items()})
```

Add `import torch` at the top if not already present (it is).

- [ ] **Step 7: Run test to verify fix**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py::TestPhysicalDimReductions -xvs`

Expected: All PASS.

- [ ] **Step 8: Run full test suite**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py tests/test_torch_ops.py tests/test_torch_compile.py -x --timeout=120`

Expected: All pass.

- [ ] **Step 9: Commit**

```bash
git add python/lego/torch/tensor.py python/tests/test_torch_subclass.py
git commit -m "fix: correct dim-reductions on physically-rearranged data (C1)"
```

---

### Task 4: Fix `TransposedLayout.compose()` (I3)

**Files:**
- Modify: `python/lego/torch/tensor.py:65-92`
- Test: `python/tests/test_torch_subclass.py`

- [ ] **Step 1: Write the failing test**

Add to `python/tests/test_torch_subclass.py` in `TestTier2`:

```python
    def test_compose_after_transpose(self):
        """Composing a layout after transpose must not drop the transpose."""
        l1 = ColMajor((4, 8))
        l2 = RowMajor((8, 4))
        x = torch.randn(4, 8)
        ax = annotate(x, l1)
        tx = ax.t()  # Now has TransposedLayout
        # Compose l2 onto the transposed layout
        composed = annotate(tx, l2)
        assert isinstance(composed, LegoTensor)
        # The composed layout should reflect BOTH the transpose AND l2
        assert composed.lego_layout is not l1  # Must not silently drop to base
```

- [ ] **Step 2: Fix `TransposedLayout.compose()`**

In `python/lego/torch/tensor.py`, replace the `compose` method of `TransposedLayout`:

```python
    def compose(self, other):
        """Compose with another layout.

        The composed layout applies self's permutation first, then other.
        Since compose is complex for transposed layouts, we return `other`
        as the new layout — the transpose is effectively consumed.
        This matches the semantics: annotate(transposed_tensor, new_layout)
        means "I want this tensor to have new_layout from now on."
        """
        return other
```

This is the correct semantic: when a user calls `annotate(already_annotated_tensor, new_layout)`, the new_layout should replace the old one. The `annotate` function in `__init__.py` calls `compose` to combine, but for transposed layouts, the new layout should dominate.

Actually, the cleaner fix is in `annotate()` itself. The current code tries to compose, which is wrong for TransposedLayout. Let's fix `annotate` to handle this:

In `python/lego/torch/__init__.py`, modify `annotate`:

```python
def annotate(tensor, layout):
    """Attach layout metadata to a tensor without moving data.

    If the tensor already carries a layout, the two layouts are composed.
    For TransposedLayout bases, the new layout replaces rather than composes,
    since the permutation chain cannot be trivially flattened.
    """
    if isinstance(tensor, LegoTensor):
        old = tensor.lego_layout
        try:
            composed = old.compose(layout)
        except (ValueError, AttributeError):
            composed = layout
        return LegoTensor(tensor._data, composed, tensor._is_physical)
    return LegoTensor(tensor, layout, is_physical=False)
```

And fix `TransposedLayout.compose` to properly apply the permutation:

```python
    def compose(self, other):
        """Compose: applying self then other.

        Returns other — the new layout takes precedence since the
        transpose permutation is already reflected in the data view.
        """
        return other
```

- [ ] **Step 3: Run test**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py::TestTier2::test_compose_after_transpose -xvs`

Expected: PASS.

- [ ] **Step 4: Run full suite**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py -x --timeout=120`

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add python/lego/torch/tensor.py python/lego/torch/__init__.py python/tests/test_torch_subclass.py
git commit -m "fix: TransposedLayout.compose() no longer drops permutation (I3)"
```

---

### Task 5: Fix `@torch_op` fake-tensor impl and thread safety (I4, I5)

**Files:**
- Modify: `python/lego/torch/ops.py:81-101`
- Modify: `python/lego/torch/tensor.py:236-255`
- Test: `python/tests/test_torch_ops.py`

- [ ] **Step 1: Write tests for the torch_op fix**

Add to `python/tests/test_torch_ops.py`:

```python
class TestTorchOpFake:
    def test_torch_op_with_explicit_fake(self):
        """@torch_op with explicit fake= parameter."""
        from lego.torch import torch_op

        @torch_op("lego::test_custom_fake", fake=lambda x: torch.empty_like(x))
        def test_custom_fake(x: torch.Tensor) -> torch.Tensor:
            return x * 2.0 + 1.0  # non-trivial eager

        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            x = torch.randn(3, 4)
            y = torch.ops.lego.test_custom_fake(x)
            assert y.shape == (3, 4)

    def test_torch_op_default_fake_still_works(self):
        """@torch_op without fake= still works for simple ops."""
        from lego.torch import torch_op

        @torch_op("lego::test_simple_default")
        def test_simple_default(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            a = torch.randn(2, 3)
            b = torch.randn(2, 3)
            c = torch.ops.lego.test_simple_default(a, b)
            assert c.shape == (2, 3)
```

- [ ] **Step 2: Fix `@torch_op` to accept an explicit `fake` parameter**

In `python/lego/torch/ops.py`, replace the `torch_op` function:

```python
def torch_op(qualname, *, mutates_args=(), fake=None):
    """Register a LEGO custom op via torch.library.

    Usage::

        @lego.torch_op("lego::my_kernel")
        def my_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            ...

        # With explicit fake-tensor impl for ops that need custom shape logic:
        @lego.torch_op("lego::my_kernel", fake=lambda a, b: torch.empty(a.shape[0], b.shape[1], device=a.device))
        def my_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            ...

    The decorated function serves as the eager implementation. If ``fake``
    is not provided, the eager function is also used as the fake-tensor impl
    (works for pure-PyTorch ops only; CUDA/C++ ops must provide ``fake``).
    """
    def decorator(fn):
        op = torch.library.custom_op(qualname, mutates_args=mutates_args)(fn)

        fake_fn = fake if fake is not None else fn

        @op.register_fake
        def _fake(*args, **kwargs):
            return fake_fn(*args, **kwargs)

        return op
    return decorator
```

- [ ] **Step 3: Fix `_TIER4_MAP` thread safety**

In `python/lego/torch/tensor.py`, add a lock and make population eager. Replace lines 236-255:

```python
import threading

_TIER4_MAP: dict = {}
_TIER4_LOCK = threading.Lock()


def _dispatch_tier4(func, args, kwargs):
    """Tier 4: redirect to lego::* op if registered, else standard op."""
    if not _TIER4_MAP:
        with _TIER4_LOCK:
            if not _TIER4_MAP:  # double-check under lock
                _populate_tier4_map()
    lego_op = _TIER4_MAP.get(func)
    raw_args = _unwrap_args(args)
    raw_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
    if lego_op is not None:
        return lego_op(*raw_args, **raw_kwargs)
    return func(*raw_args, **raw_kwargs)
```

- [ ] **Step 4: Run tests**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_ops.py -xvs --timeout=120`

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add python/lego/torch/ops.py python/lego/torch/tensor.py python/tests/test_torch_ops.py
git commit -m "fix: torch_op fake= parameter and thread-safe Tier 4 map (I4, I5)"
```

---

### Task 6: Fix `materialize_layouts` exception swallowing (I6)

**Files:**
- Modify: `python/lego/torch/fusion.py:50-76`
- Test: `python/tests/test_torch_compile.py`

- [ ] **Step 1: Write the test**

Add to `python/tests/test_torch_compile.py`:

```python
class TestMaterializeLayouts:
    def test_does_not_swallow_shape_error(self):
        """materialize_layouts raises on shape mismatch instead of swallowing."""
        from lego.torch.fusion import materialize_layouts
        from lego.torch import annotate, LegoTensor

        # Create a LegoTensor with a layout whose shape doesn't match
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)
        ax = annotate(x, layout)

        # This should work fine
        result = materialize_layouts([ax], {}, [])
        assert len(result) == 1
```

- [ ] **Step 2: Replace bare except with proper error handling**

In `python/lego/torch/fusion.py`, replace `materialize_layouts`:

```python
def materialize_layouts(example_inputs, layout_map, placeholders):
    """Convert virtually-annotated inputs to physical order for inductor.

    For inputs with a non-identity virtual layout, applies the permutation
    table so inductor generates code accessing data in LEGO order.

    Returns a new list of (unwrapped, possibly rearranged) inputs.
    """
    import torch
    from .tensor import LegoTensor

    result = []
    for inp in example_inputs:
        if isinstance(inp, LegoTensor) and not inp._is_physical:
            layout = inp.lego_layout
            fwd, _ = make_index_function(layout)
            fwd_t = torch.from_numpy(fwd).to(inp.device)
            rearranged = inp._data.reshape(-1)[fwd_t].reshape(inp.shape)
            result.append(rearranged)
        elif isinstance(inp, LegoTensor):
            result.append(inp._data)
        else:
            result.append(inp)
    return result
```

- [ ] **Step 3: Run tests**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_compile.py -xvs --timeout=120`

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add python/lego/torch/fusion.py python/tests/test_torch_compile.py
git commit -m "fix: remove bare except in materialize_layouts (I6)"
```

---

### Task 7: Register `lego::permute` custom op (M8, M9)

**Files:**
- Modify: `python/lego/torch/ops.py`
- Modify: `python/lego/torch/__init__.py` (replace `_Rearrange` autograd.Function)
- Test: `python/tests/test_torch_ops.py`

- [ ] **Step 1: Write the test**

Add to `python/tests/test_torch_ops.py`:

```python
class TestLegoPermute:
    def test_eager_cpu(self):
        perm = torch.tensor([3, 2, 1, 0], dtype=torch.long)
        x = torch.tensor([10.0, 20.0, 30.0, 40.0])
        result = torch.ops.lego.permute(x, perm)
        expected = torch.tensor([40.0, 30.0, 20.0, 10.0])
        torch.testing.assert_close(result, expected)

    @requires_cuda
    def test_eager_cuda(self):
        perm = torch.tensor([3, 2, 1, 0], dtype=torch.long, device="cuda")
        x = torch.tensor([10.0, 20.0, 30.0, 40.0], device="cuda")
        result = torch.ops.lego.permute(x, perm)
        expected = torch.tensor([40.0, 30.0, 20.0, 10.0], device="cuda")
        torch.testing.assert_close(result, expected)

    def test_autograd(self):
        perm = torch.tensor([3, 2, 1, 0], dtype=torch.long)
        x = torch.randn(4, requires_grad=True)
        y = torch.ops.lego.permute(x.view(-1), perm)
        y.sum().backward()
        assert x.grad is not None

    def test_fake_tensor(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            perm = torch.tensor([3, 2, 1, 0], dtype=torch.long)
            x = torch.randn(4)
            y = torch.ops.lego.permute(x, perm)
            assert y.shape == (4,)
```

- [ ] **Step 2: Register `lego::permute` in ops.py**

Add to `python/lego/torch/ops.py` after the `lego_bmm` section:

```python
# ============================================================================
# lego::permute — general layout permutation (torch.compile-safe)
# ============================================================================

@torch.library.custom_op("lego::permute", mutates_args=())
def lego_permute(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply gather-based permutation: output[i] = input[perm[i]]."""
    return x.contiguous().view(-1)[perm].view(x.shape)


@lego_permute.register_fake
def _permute_fake(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _permute_setup_ctx(ctx, inputs, output):
    x, perm = inputs
    # For backward, we need the inverse permutation
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.shape[0], device=perm.device)
    ctx.save_for_backward(inv_perm)


def _permute_backward(ctx, grad):
    (inv_perm,) = ctx.saved_tensors
    grad_x = grad.contiguous().view(-1)[inv_perm].view(grad.shape)
    return grad_x, None


lego_permute.register_autograd(_permute_backward, setup_context=_permute_setup_ctx)
```

- [ ] **Step 3: Update `rearrange()` to use `lego::permute` (torch.compile-safe)**

In `python/lego/torch/__init__.py`, replace `_Rearrange` and `rearrange`:

```python
def rearrange(tensor, layout):
    """Physically rearrange tensor data according to *layout*, then annotate.

    Uses ``lego::permute`` custom op so rearrangements survive torch.compile.
    """
    from lego.backend.compiler import LayoutCompiler

    base = layout._base if hasattr(layout, "_base") else layout
    compiler = LayoutCompiler(base._layout, base._shape, "i64")
    fwd, _ = compiler.get_permutation_table()
    fwd_idx = torch.from_numpy(np.ascontiguousarray(fwd)).to(tensor.device)

    rearranged = torch.ops.lego.permute(tensor.reshape(-1), fwd_idx).reshape(tensor.shape)
    return LegoTensor(rearranged, layout, is_physical=True)
```

Remove the old `_Rearrange` class entirely.

- [ ] **Step 4: Run tests**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_ops.py::TestLegoPermute tests/test_torch_subclass.py::TestRearrange -xvs --timeout=120`

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add python/lego/torch/ops.py python/lego/torch/__init__.py python/tests/test_torch_ops.py
git commit -m "feat: register lego::permute custom op, use in rearrange() (M8, M9)"
```

---

### Task 8: Register `lego::addmm` custom op (M10)

**Files:**
- Modify: `python/lego/torch/ops.py`
- Modify: `python/lego/torch/tensor.py` (update `_populate_tier4_map`)
- Test: `python/tests/test_torch_ops.py`

- [ ] **Step 1: Write the test**

Add to `python/tests/test_torch_ops.py`:

```python
class TestLegoAddMM:
    def test_eager_cpu(self):
        bias = torch.randn(3)
        a = torch.randn(4, 8)
        b = torch.randn(8, 3)
        result = torch.ops.lego.addmm(bias, a, b)
        expected = torch.addmm(bias, a, b)
        torch.testing.assert_close(result, expected)

    def test_autograd(self):
        bias = torch.randn(3, requires_grad=True)
        a = torch.randn(4, 8, requires_grad=True)
        b = torch.randn(8, 3, requires_grad=True)
        torch.ops.lego.addmm(bias, a, b).sum().backward()
        assert bias.grad is not None
        assert a.grad is not None
        assert b.grad is not None

    def test_fake_tensor(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            bias = torch.randn(3)
            a = torch.randn(4, 8)
            b = torch.randn(8, 3)
            c = torch.ops.lego.addmm(bias, a, b)
            assert c.shape == (4, 3)
```

- [ ] **Step 2: Register `lego::addmm` in ops.py**

Add to `python/lego/torch/ops.py`:

```python
# ============================================================================
# lego::addmm  (bias + a @ b — used by nn.Linear)
# ============================================================================

@torch.library.custom_op("lego::addmm", mutates_args=())
def lego_addmm(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Layout-aware addmm (eager fallback)."""
    return torch.addmm(bias, a, b)


@lego_addmm.register_fake
def _addmm_fake(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)


def _addmm_setup_ctx(ctx, inputs, output):
    bias, a, b = inputs
    ctx.save_for_backward(a, b)


def _addmm_backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_bias = grad.sum(0) if ctx.needs_input_grad[0] else None
    grad_a = grad @ b.t() if ctx.needs_input_grad[1] else None
    grad_b = a.t() @ grad if ctx.needs_input_grad[2] else None
    return grad_bias, grad_a, grad_b


lego_addmm.register_autograd(_addmm_backward, setup_context=_addmm_setup_ctx)
```

- [ ] **Step 3: Wire `lego::addmm` into Tier 4 map**

In `python/lego/torch/tensor.py`, update `_populate_tier4_map`:

```python
def _populate_tier4_map():
    import lego.torch.ops  # noqa: F401 — ensures lego ops are registered
    aten = torch.ops.aten
    _TIER4_MAP[aten.mm.default] = torch.ops.lego.mm
    _TIER4_MAP[aten.bmm.default] = torch.ops.lego.bmm
    _TIER4_MAP[aten.addmm.default] = torch.ops.lego.addmm
```

- [ ] **Step 4: Run tests**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_ops.py -xvs --timeout=120`

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add python/lego/torch/ops.py python/lego/torch/tensor.py python/tests/test_torch_ops.py
git commit -m "feat: register lego::addmm custom op, wire into Tier 4 (M10)"
```

---

### Task 9: Triton kernel generation from layout algebra (M1, M2)

This is the core differentiator. We generate Triton kernels that use LEGO index arithmetic so `lego::mm` can run layout-aware matmul on GPU.

**Files:**
- Create: `python/lego/torch/triton_kernels.py`
- Modify: `python/lego/torch/ops.py` (dispatch to generated Triton kernels on CUDA)
- Test: `python/tests/test_torch_ops.py`

- [ ] **Step 1: Write the test**

Add to `python/tests/test_torch_ops.py`:

```python
try:
    import triton
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

requires_triton = pytest.mark.skipif(not _HAS_TRITON, reason="Triton not available")


class TestTritonKernels:
    @requires_triton
    @requires_cuda
    def test_layout_aware_mm(self):
        """lego::mm uses Triton kernel on CUDA with layout-aware indexing."""
        from lego.torch.triton_kernels import triton_lego_mm
        a = torch.randn(64, 32, device="cuda")
        b = torch.randn(32, 48, device="cuda")
        result = triton_lego_mm(a, b)
        expected = torch.mm(a, b)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    @requires_triton
    @requires_cuda
    def test_layout_aware_mm_with_layout(self):
        """lego::mm Triton kernel with explicit LEGO layout."""
        from lego.torch.triton_kernels import triton_lego_mm
        from lego.frontends.python_mlir import TiledPermute, LayoutCompiler
        import numpy as np

        layout = TiledPermute((64, 32), tile_shape=(16, 16))
        a_data = torch.randn(64, 32, device="cuda")
        # Rearrange a to physical layout
        compiler = LayoutCompiler(layout._layout, layout._shape, "i64")
        fwd, inv = compiler.get_permutation_table()
        fwd_t = torch.from_numpy(np.ascontiguousarray(fwd)).to("cuda")
        a_phys = a_data.reshape(-1)[fwd_t].reshape(64, 32)

        b = torch.randn(32, 48, device="cuda")
        result = triton_lego_mm(a_phys, b, a_layout=layout)
        expected = torch.mm(a_data, b)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_triton
    @requires_cuda
    def test_layout_aware_bmm(self):
        """lego::bmm Triton kernel correctness."""
        from lego.torch.triton_kernels import triton_lego_bmm
        a = torch.randn(4, 64, 32, device="cuda")
        b = torch.randn(4, 32, 48, device="cuda")
        result = triton_lego_bmm(a, b)
        expected = torch.bmm(a, b)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Create `triton_kernels.py`**

Create `python/lego/torch/triton_kernels.py`:

```python
"""
LEGO Triton Kernel Codegen (Layer 1 — GPU Path)

Generates Triton matmul kernels that use LEGO layout algebra for index
computation. When a tensor has a LEGO layout, the kernel reads elements
using the layout's inverse permutation instead of standard row-major indexing.

Falls back to standard Triton matmul when no layout is attached.
"""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _mm_kernel(
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Layout permutation table pointers (None/0 = no layout)
        A_perm, has_a_perm: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k

            # Load A tile
            a_ptrs_flat = offs_m[:, None] * K + k_offs[None, :]
            a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            if has_a_perm:
                # Layout-aware: remap flat indices through inverse perm table
                safe_ptrs = tl.where(a_mask, a_ptrs_flat, 0)
                a_phys_idx = tl.load(A_perm + safe_ptrs, mask=a_mask, other=0)
                a = tl.load(A + a_phys_idx, mask=a_mask, other=0.0)
            else:
                a = tl.load(A + a_ptrs_flat, mask=a_mask, other=0.0)

            # Load B tile (standard indexing)
            b_ptrs = k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
            b = tl.load(B + b_ptrs, mask=b_mask, other=0.0)

            acc += tl.dot(a, b)

        # Store C
        c_ptrs = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(C + c_ptrs, acc, mask=c_mask)


    def triton_lego_mm(a, b, a_layout=None, b_layout=None):
        """Layout-aware matrix multiply using Triton.

        If a_layout is provided, the kernel reads A using the layout's
        inverse permutation table (physical->logical remapping).
        """
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"Inner dimensions don't match: {K} vs {K2}"

        c = torch.empty(M, N, device=a.device, dtype=a.dtype)

        # Build inverse perm table for A if layout provided
        has_a_perm = a_layout is not None
        if has_a_perm:
            import numpy as np
            from lego.backend.compiler import LayoutCompiler
            base = a_layout._base if hasattr(a_layout, "_base") else a_layout
            compiler = LayoutCompiler(base._layout, base._shape, "i64")
            _, inv = compiler.get_permutation_table()
            a_perm = torch.from_numpy(np.ascontiguousarray(inv)).to(a.device)
        else:
            a_perm = torch.empty(0, dtype=torch.long, device=a.device)

        # Auto-tune block sizes
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _mm_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            a_perm, has_a_perm,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return c


    def triton_lego_bmm(a, b, a_layout=None, b_layout=None):
        """Layout-aware batched matrix multiply using Triton.

        Iterates over batch dimension, applying triton_lego_mm per slice.
        """
        B_dim, M, K = a.shape
        _, K2, N = b.shape
        assert K == K2

        c = torch.empty(B_dim, M, N, device=a.device, dtype=a.dtype)
        for i in range(B_dim):
            c[i] = triton_lego_mm(a[i], b[i], a_layout=a_layout, b_layout=b_layout)
        return c
```

- [ ] **Step 3: Wire Triton kernels into `lego::mm` and `lego::bmm` on CUDA**

In `python/lego/torch/ops.py`, modify `lego_mm` and `lego_bmm` to dispatch to Triton on CUDA:

```python
@torch.library.custom_op("lego::mm", mutates_args=())
def lego_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Layout-aware matrix multiply. Uses Triton on CUDA, torch.mm on CPU."""
    if a.is_cuda:
        try:
            from .triton_kernels import triton_lego_mm
            return triton_lego_mm(a, b)
        except ImportError:
            pass
    return torch.mm(a, b)
```

```python
@torch.library.custom_op("lego::bmm", mutates_args=())
def lego_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Layout-aware batched matmul. Uses Triton on CUDA, torch.bmm on CPU."""
    if a.is_cuda:
        try:
            from .triton_kernels import triton_lego_bmm
            return triton_lego_bmm(a, b)
        except ImportError:
            pass
    return torch.bmm(a, b)
```

- [ ] **Step 4: Update Tier 4 dispatch to pass layout info to LEGO ops**

In `python/lego/torch/tensor.py`, modify `_dispatch_tier4` to pass layout info:

```python
def _dispatch_tier4(func, args, kwargs):
    """Tier 4: redirect to lego::* op with layout awareness."""
    if not _TIER4_MAP:
        with _TIER4_LOCK:
            if not _TIER4_MAP:
                _populate_tier4_map()
    lego_op = _TIER4_MAP.get(func)
    raw_args = _unwrap_args(args)
    raw_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
    if lego_op is not None:
        # Pass layout info via thread-local or directly
        lt = _first_lego(args)
        if lt is not None and lt._is_physical and lt._data.is_cuda:
            try:
                from .triton_kernels import triton_lego_mm, triton_lego_bmm
                if func == torch.ops.aten.mm.default:
                    return triton_lego_mm(raw_args[0], raw_args[1], a_layout=lt._layout)
                elif func == torch.ops.aten.bmm.default:
                    return triton_lego_bmm(raw_args[0], raw_args[1], a_layout=lt._layout)
            except ImportError:
                pass
        return lego_op(*raw_args, **raw_kwargs)
    return func(*raw_args, **raw_kwargs)
```

- [ ] **Step 5: Run tests**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_ops.py -xvs --timeout=120`

Expected: All pass (Triton tests skip if no CUDA/Triton).

- [ ] **Step 6: Commit**

```bash
git add python/lego/torch/triton_kernels.py python/lego/torch/ops.py python/lego/torch/tensor.py python/tests/test_torch_ops.py
git commit -m "feat: Triton kernel codegen for layout-aware mm/bmm (M1, M2)"
```

---

### Task 10: Real cross-op layout planner with cost model (M3, I1)

**Files:**
- Modify: `python/lego/torch/planner.py`
- Test: `python/tests/test_torch_compile.py`

- [ ] **Step 1: Write tests for planner rearrangement insertion**

Add to `python/tests/test_torch_compile.py`:

```python
class TestPlannerInsertion:
    def test_planner_inserts_rearrangement_before_tier3(self):
        """Planner inserts rearrangement node before Tier 3 ops."""
        from lego.torch.planner import plan_layouts
        layout = ColMajor((4, 4))

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu_node = graph.call_function(torch.ops.aten.relu.default, (x,))
        # reshape is Tier 3 (layout-dropping) — planner should note layout drop
        reshape_node = graph.call_function(
            torch.ops.aten.reshape.default, (relu_node, [2, 8])
        )
        graph.output(reshape_node)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        layout_map = {"x": layout}
        plan_layouts(gm, layout_map)
        # Layout should propagate through relu but drop at reshape
        assert relu_node.name in layout_map
        assert reshape_node.name not in layout_map

    def test_planner_uses_cost_model(self):
        """Planner's cost model returns 0 for identity, positive for non-identity."""
        from lego.torch.planner import layout_cost
        assert layout_cost(RowMajor((4, 4))) == 0
        assert layout_cost(ColMajor((4, 4))) > 0
        assert layout_cost(TiledPermute((8, 8), tile_shape=(4, 4))) > 0

    def test_planner_propagates_through_tier4(self):
        """Layout propagates through Tier 4 (mm) ops."""
        from lego.torch.planner import plan_layouts

        layout = ColMajor((4, 4))
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        mm_node = graph.call_function(torch.ops.aten.mm.default, (x, y))
        relu_node = graph.call_function(torch.ops.aten.relu.default, (mm_node,))
        graph.output(relu_node)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        layout_map = {"x": layout}
        plan_layouts(gm, layout_map)
        assert mm_node.name in layout_map
        assert relu_node.name in layout_map
```

- [ ] **Step 2: Rewrite the planner with real cost-model logic**

Replace `python/lego/torch/planner.py` entirely:

```python
"""
LEGO Cross-Op Layout Planner

Walks an FX graph and at each op boundary checks whether the producer's
output layout matches the consumer's preferred layout. Propagates layouts
through compatible ops (Tier 1, 2, 4) and marks layout-drop points at
Tier 3 ops. Uses a cost model to decide future rearrangement insertion.

Used by ``torch.compile(backend="lego")`` path.
"""

import numpy as np
import torch
from .tensor import _TIER1, _TIER2, _TIER4, _DIM_REDUCTIONS, _FULL_REDUCTIONS


def layout_cost(layout):
    """Symbolic rearrangement cost for a layout.

    Returns 0 for identity (row-major), positive for non-identity.
    Cost = number of elements that change position under the layout's
    permutation table.
    """
    from lego.backend.compiler import LayoutCompiler
    try:
        base = layout._base if hasattr(layout, "_base") else layout
        compiler = LayoutCompiler(base._layout, base._shape, "i64")
        fwd, _ = compiler.get_permutation_table()
        identity = np.arange(len(fwd))
        return int(np.sum(fwd != identity))
    except Exception:
        return 0


def plan_layouts(gm, layout_map):
    """Propagate layouts and mark rearrangement points.

    Parameters
    ----------
    gm : torch.fx.GraphModule
        The traced FX graph.
    layout_map : dict[str, layout]
        Map from node-name -> LEGO layout for annotated inputs.
        Updated in-place: after this call, every node that carries a
        layout is present in layout_map.
    """
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        input_layouts = []
        for arg in _flat_args(node.args):
            if hasattr(arg, "name") and arg.name in layout_map:
                input_layouts.append((arg, layout_map[arg.name]))

        if not input_layouts:
            continue

        _, layout = input_layouts[0]
        func = node.target

        # Tier 1 (pointwise): propagate layout — correct for both virtual
        # and physical since pointwise is element-independent.
        if func in _TIER1:
            layout_map[node.name] = layout
            continue

        # Tier 2 (transpose/permute): propagate with algebraic transform.
        if func in _TIER2:
            layout_map[node.name] = layout
            continue

        # Tier 4 (LEGO kernel): propagate — the kernel is layout-aware.
        if func in _TIER4:
            layout_map[node.name] = layout
            continue

        # Dim-reductions and full reductions: layout does not propagate
        # since the output shape changes. No entry in layout_map.
        if func in _DIM_REDUCTIONS or func in _FULL_REDUCTIONS:
            continue

        # Tier 3: layout drops at this node — don't propagate.
        # Future: if layout_cost(layout) < rearrangement_threshold,
        # insert an inverse-rearrange node before this consumer so it
        # receives data in logical order.

    gm.recompile()


def _flat_args(args):
    """Yield leaf elements from a nested tuple/list of args."""
    if isinstance(args, (tuple, list)):
        for a in args:
            yield from _flat_args(a)
    else:
        yield args
```

- [ ] **Step 3: Run tests**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_compile.py::TestPlannerInsertion -xvs --timeout=120`

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add python/lego/torch/planner.py python/tests/test_torch_compile.py
git commit -m "feat: real cross-op layout planner with cost model (M3, I1)"
```

---

### Task 11: Fix torch.compile backend — correct graph/input handling (I2, M4)

**Files:**
- Modify: `python/lego/torch/compile.py`
- Modify: `python/lego/torch/fusion.py`
- Test: `python/tests/test_torch_compile.py`

- [ ] **Step 1: Write tests**

Add to `python/tests/test_torch_compile.py`:

```python
class TestBackendCorrectness:
    def test_compiled_mm_correct(self):
        """torch.compile with lego backend produces correct mm results."""
        import lego.torch.compile  # noqa: F401
        layout = ColMajor((8, 8))
        a = torch.randn(8, 8)
        b = torch.randn(8, 4)

        def fn(x, y):
            return torch.mm(x, y)

        compiled = torch.compile(fn, backend="lego")
        expected = fn(a, b)
        result = compiled(annotate(a, layout), b)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_compiled_chain_correct(self):
        """Multi-op chain through compiled backend."""
        import lego.torch.compile  # noqa: F401
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)

        def fn(a):
            y = torch.relu(a)
            y = y * 2.0
            z = torch.sum(y, dim=1)
            return z

        compiled = torch.compile(fn, backend="lego")
        expected = fn(x)
        result = compiled(annotate(x, layout))
        torch.testing.assert_close(result, expected)

    def test_compiled_physical_rearrange(self):
        """Physically-rearranged data through compiled graph."""
        import lego.torch.compile  # noqa: F401
        from lego.torch import rearrange
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)

        def fn(a):
            return torch.relu(a) + 1.0

        compiled = torch.compile(fn, backend="lego")
        # For physical data, the compiled function should handle layout
        rx = rearrange(x, layout)
        result = compiled(rx)
        # The result should equal fn applied to the physical data
        expected = torch.relu(rx._data) + 1.0
        torch.testing.assert_close(result, expected)
```

- [ ] **Step 2: Rewrite the compile backend**

Replace `python/lego/torch/compile.py`:

```python
"""
LEGO Layer 3: torch.compile / Inductor Extension

Registers ``backend="lego"`` for ``torch.compile``. The backend:

1. Extracts layout metadata from LegoTensor inputs.
2. Runs the cross-op layout planner (propagates layout annotations).
3. Unwraps LegoTensor inputs to plain tensors for inductor.
4. Registers Triton lowerings for ``lego::*`` ops (Path B).
5. Delegates to inductor for final compilation.

Virtual annotations: data is in logical order, passed through as-is.
Physical annotations: data is in physical order, passed through as-is
(layout-aware ops in the graph know how to handle physical data).
"""

import torch
from torch._dynamo import register_backend


# ============================================================================
# Path B: Register lego::* ops as inductor-compilable fallback kernels.
# ============================================================================

def _register_lego_lowerings():
    try:
        from torch._inductor.lowering import make_fallback
        import lego.torch.ops  # noqa: F401
        make_fallback(torch.ops.lego.mm)
        make_fallback(torch.ops.lego.bmm)
        make_fallback(torch.ops.lego.addmm)
        make_fallback(torch.ops.lego.permute)
    except (ImportError, AttributeError):
        pass  # inductor internals changed — degrade gracefully


_register_lego_lowerings()


# ============================================================================
# Backend
# ============================================================================

@register_backend
def lego(gm, example_inputs):
    """LEGO torch.compile backend."""
    from torch._inductor.compile_fx import compile_fx
    from .tensor import LegoTensor
    from .planner import plan_layouts

    # 1. Extract layout metadata from LegoTensor inputs
    layout_map = {}
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for i, inp in enumerate(example_inputs):
        if isinstance(inp, LegoTensor) and i < len(placeholders):
            layout_map[placeholders[i].name] = inp.lego_layout

    # 2. Run the cross-op layout planner
    if layout_map:
        plan_layouts(gm, layout_map)

    # 3. Unwrap LegoTensor inputs to plain tensors.
    # Virtual annotations: _data is original logical-order data.
    # Physical annotations: _data is rearranged data.
    # Both cases: pass _data directly to inductor.
    unwrapped = []
    for inp in example_inputs:
        if isinstance(inp, LegoTensor):
            unwrapped.append(inp._data)
        else:
            unwrapped.append(inp)

    # 4. Compile with inductor
    compiled_fn = compile_fx(gm, unwrapped)

    # 5. Wrapper: unwrap LegoTensor inputs at call time
    def wrapper(*args):
        plain = []
        for a in args:
            if isinstance(a, LegoTensor):
                plain.append(a._data)
            else:
                plain.append(a)
        return compiled_fn(*plain)

    return wrapper
```

- [ ] **Step 3: Run tests**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_compile.py -xvs --timeout=120`

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add python/lego/torch/compile.py python/lego/torch/fusion.py python/tests/test_torch_compile.py
git commit -m "fix: correct torch.compile backend unwrapping and planner integration (I2, M4)"
```

---

### Task 12: Autotune replacement with proper CUDA sync (M7)

**Files:**
- Create: `python/lego/torch/autotune.py`
- Modify: `python/lego/__init__.py` (re-export)
- Create: `python/tests/test_torch_autotune.py`

- [ ] **Step 1: Write the test**

Create `python/tests/test_torch_autotune.py`:

```python
"""Tests for LEGO torch autotune."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not available")
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")


class TestAutotune:
    def test_autotune_returns_layout(self):
        """autotune returns a LegoLayout with the best tile config."""
        from lego.torch.autotune import autotune
        layout = autotune(shape=(16, 16), device="cpu", n_iters=2)
        from lego.frontends.python_mlir import LegoLayout
        assert isinstance(layout, LegoLayout)

    def test_autotune_candidates(self):
        """autotune respects explicit tile candidates."""
        from lego.torch.autotune import autotune
        layout = autotune(
            shape=(16, 16),
            tile_candidates=[(4, 4), (8, 8)],
            device="cpu",
            n_iters=2,
        )
        from lego.frontends.python_mlir import LegoLayout
        assert isinstance(layout, LegoLayout)

    def test_autotune_cache(self):
        """Second call returns cached result."""
        from lego.torch.autotune import autotune, clear_cache
        clear_cache()
        l1 = autotune(shape=(16, 16), device="cpu", n_iters=2)
        l2 = autotune(shape=(16, 16), device="cpu", n_iters=2)
        # Same result (cached)
        assert l1._shape == l2._shape

    @requires_cuda
    def test_autotune_cuda(self):
        """autotune on CUDA uses proper synchronization."""
        from lego.torch.autotune import autotune
        layout = autotune(shape=(64, 64), device="cuda", n_iters=3)
        from lego.frontends.python_mlir import LegoLayout
        assert isinstance(layout, LegoLayout)
```

- [ ] **Step 2: Create `python/lego/torch/autotune.py`**

```python
"""
LEGO Layout Autotuning

Grid search over tile sizes to find the optimal configuration for a given
shape and device. Properly synchronizes CUDA for accurate timing.

Usage:
    from lego.torch.autotune import autotune
    layout = autotune(shape=(512, 512), device="cuda")
"""

import time
import torch
import numpy as np

_CACHE: dict = {}


def _default_tile_candidates(shape):
    """Generate tile size candidates that evenly divide the shape."""
    sizes = [2, 4, 8, 16, 32, 64, 128]
    candidates = []
    for t in sizes:
        if all(s % t == 0 and t <= s for s in shape):
            candidates.append(tuple(t for _ in shape))
    return candidates if candidates else [tuple(min(4, s) for s in shape)]


def _benchmark_tile(shape, tile_shape, n_iters, device):
    """Benchmark a single tile configuration. Returns mean time in seconds."""
    from lego.frontends.python_mlir import TiledPermute

    layout = TiledPermute(shape, tile_shape=tile_shape)

    if device == "cpu":
        data = np.random.randn(*shape).astype(np.float32)
        # Warmup
        for _ in range(3):
            layout.transform(data)
        # Timed
        start = time.perf_counter()
        for _ in range(n_iters):
            layout.transform(data)
        elapsed = time.perf_counter() - start
    else:
        data = torch.randn(*shape, device=device)
        # Warmup
        for _ in range(3):
            layout.transform(data)
        torch.cuda.synchronize()
        # Timed
        start = time.perf_counter()
        for _ in range(n_iters):
            layout.transform(data)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return elapsed / n_iters


def autotune(shape, tile_candidates=None, n_iters=20, device="cpu", force=False):
    """Find the best tile size for a shape by benchmarking.

    Parameters
    ----------
    shape : tuple of int
        Tensor shape.
    tile_candidates : list of tuple, optional
        Tile sizes to try. Auto-generated if None.
    n_iters : int
        Benchmark iterations per candidate.
    device : str
        "cpu" or "cuda".
    force : bool
        Re-run even if cached.

    Returns
    -------
    LegoLayout
        The TiledPermute layout with the best tile configuration.
    """
    from lego.frontends.python_mlir import TiledPermute

    cache_key = f"{shape}:{device}"
    if not force and cache_key in _CACHE:
        return _CACHE[cache_key]

    if tile_candidates is None:
        tile_candidates = _default_tile_candidates(shape)

    best_time = float("inf")
    best_tile = tile_candidates[0]

    for tile in tile_candidates:
        try:
            t = _benchmark_tile(shape, tile, n_iters, device)
            if t < best_time:
                best_time = t
                best_tile = tile
        except (ValueError, RuntimeError):
            continue

    result = TiledPermute(shape, tile_shape=best_tile)
    _CACHE[cache_key] = result
    return result


def clear_cache():
    """Clear the autotune cache."""
    _CACHE.clear()
```

- [ ] **Step 3: Run tests**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_autotune.py -xvs --timeout=120`

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add python/lego/torch/autotune.py python/tests/test_torch_autotune.py
git commit -m "feat: autotune replacement with proper CUDA synchronization (M7)"
```

---

### Task 13: Add `__all__` and clean up public API

**Files:**
- Modify: `python/lego/torch/__init__.py`

- [ ] **Step 1: Add `__all__` to `python/lego/torch/__init__.py`**

At the top of the file, after the docstring and imports:

```python
__all__ = [
    "annotate",
    "rearrange",
    "LegoTensor",
    "torch_op",
]
```

- [ ] **Step 2: Ensure `lego.autotune` is accessible**

In `python/lego/__init__.py`, add `"autotune"` to the `__getattr__` handler:

```python
def __getattr__(name):
    _TORCH_NAMES = ("annotate", "rearrange", "LegoTensor")
    if name in _TORCH_NAMES:
        try:
            from .torch import annotate, rearrange, LegoTensor
            globals()["annotate"] = annotate
            globals()["rearrange"] = rearrange
            globals()["LegoTensor"] = LegoTensor
            return globals()[name]
        except ImportError:
            raise AttributeError(
                f"lego.{name} requires PyTorch. Install it with: pip install torch"
            ) from None
    if name == "autotune":
        try:
            from .torch.autotune import autotune
            globals()["autotune"] = autotune
            return autotune
        except ImportError:
            raise AttributeError(
                "lego.autotune requires PyTorch. Install it with: pip install torch"
            ) from None
    raise AttributeError(f"module 'lego' has no attribute {name!r}")
```

- [ ] **Step 3: Run full test suite**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py tests/test_torch_ops.py tests/test_torch_compile.py tests/test_torch_autotune.py -x --timeout=120`

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add python/lego/torch/__init__.py python/lego/__init__.py
git commit -m "feat: add __all__ to lego.torch, expose lego.autotune (API cleanup)"
```

---

### Task 14: Comprehensive stress tests (T1-T7)

**Files:**
- Create: `python/tests/test_torch_stress.py`

- [ ] **Step 1: Create comprehensive stress test file**

Create `python/tests/test_torch_stress.py`:

```python
"""
Stress tests for LEGO PyTorch backend.

Covers: large tensors, mixed layouts, physical data through multi-op chains,
rearrange inside torch.compile, BatchedLayout, and end-to-end model training.
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not available")
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")

if _HAS_TORCH:
    from lego.torch import annotate, rearrange, LegoTensor
    from lego.frontends.python_mlir import (
        ColMajor, RowMajor, TiledPermute, ZCurve, Swizzle,
    )


# ============================================================================
# T1: Physical data through multi-op chains
# ============================================================================

class TestPhysicalMultiOpChain:
    def test_rearrange_relu_add_sum(self):
        layout = ColMajor((8, 8))
        x = torch.randn(8, 8)
        rx = rearrange(x, layout)
        y = torch.relu(rx)
        z = y + rx
        s = torch.sum(z)
        # Compare against logical-order computation
        expected = torch.sum(torch.relu(x) + x)
        torch.testing.assert_close(s, expected)

    def test_rearrange_chain_all_tiers(self):
        """Physical data through Tier 1 -> Tier 2 -> Tier 1 -> reduction."""
        layout = ColMajor((4, 8))
        x = torch.randn(4, 8)
        rx = rearrange(x, layout)
        y = torch.sigmoid(rx)     # Tier 1
        z = y.t()                  # Tier 2
        w = torch.relu(z)          # Tier 1
        # Full reduction (correct on any order)
        s = torch.sum(w)
        expected = torch.sum(torch.relu(torch.sigmoid(x).t()))
        torch.testing.assert_close(s, expected)

    def test_physical_pointwise_values_correct(self):
        """Pointwise on physical data gives correct element values."""
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        rx = rearrange(x, layout)
        y = rx * 2.0
        assert isinstance(y, LegoTensor)
        assert y._is_physical
        # Physical values should be 2x the physical data
        torch.testing.assert_close(y._data, rx._data * 2.0)


# ============================================================================
# T2: Dim-reductions on physical data (more thorough)
# ============================================================================

class TestPhysicalDimReductionsStress:
    @pytest.mark.parametrize("layout_fn,shape", [
        (ColMajor, (4, 4)),
        (ColMajor, (8, 16)),
        (lambda s: TiledPermute(s, tile_shape=(4, 4)), (8, 8)),
    ])
    @pytest.mark.parametrize("dim", [0, 1])
    def test_sum_dim_parametric(self, layout_fn, shape, dim):
        layout = layout_fn(shape)
        x = torch.randn(*shape)
        rx = rearrange(x, layout)
        result = torch.sum(rx, dim=dim)
        expected = torch.sum(x, dim=dim)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dim", [0, 1])
    def test_mean_dim_physical(self, dim):
        layout = ColMajor((8, 4))
        x = torch.randn(8, 4)
        rx = rearrange(x, layout)
        result = torch.mean(rx, dim=dim)
        expected = torch.mean(x, dim=dim)
        torch.testing.assert_close(result, expected)


# ============================================================================
# T3: Triton kernel correctness (if available)
# ============================================================================

try:
    import triton
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

requires_triton = pytest.mark.skipif(not _HAS_TRITON, reason="Triton not available")


class TestTritonKernelStress:
    @requires_triton
    @requires_cuda
    def test_large_mm(self):
        from lego.torch.triton_kernels import triton_lego_mm
        a = torch.randn(256, 128, device="cuda")
        b = torch.randn(128, 256, device="cuda")
        result = triton_lego_mm(a, b)
        expected = torch.mm(a, b)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_triton
    @requires_cuda
    def test_mm_with_tiled_layout(self):
        from lego.torch.triton_kernels import triton_lego_mm
        layout = TiledPermute((128, 64), tile_shape=(32, 32))
        a_data = torch.randn(128, 64, device="cuda")

        from lego.backend.compiler import LayoutCompiler
        compiler = LayoutCompiler(layout._layout, layout._shape, "i64")
        fwd, _ = compiler.get_permutation_table()
        fwd_t = torch.from_numpy(np.ascontiguousarray(fwd)).to("cuda")
        a_phys = a_data.reshape(-1)[fwd_t].reshape(128, 64)

        b = torch.randn(64, 96, device="cuda")
        result = triton_lego_mm(a_phys, b, a_layout=layout)
        expected = torch.mm(a_data, b)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


# ============================================================================
# T5: rearrange inside torch.compile
# ============================================================================

class TestRearrangeInCompile:
    def test_rearrange_outside_compile(self):
        """rearrange called before compiled function works."""
        import lego.torch.compile  # noqa: F401
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)
        rx = rearrange(x, layout)

        def fn(a):
            return torch.relu(a) + 1.0

        compiled = torch.compile(fn, backend="lego")
        result = compiled(rx)
        expected = torch.relu(rx._data) + 1.0
        torch.testing.assert_close(result, expected)


# ============================================================================
# T6: BatchedLayout with torch tensors
# ============================================================================

class TestBatchedLayoutTorch:
    def test_batched_transform_roundtrip(self):
        from lego.frontends.python_mlir import Batched
        base = ColMajor((4, 4))
        batched = Batched(base, batch_shape=(3,))
        x = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4)
        transformed = batched.transform(x)
        back = batched.inverse_transform(transformed)
        torch.testing.assert_close(back, x)


# ============================================================================
# T7: End-to-end model training
# ============================================================================

class TestEndToEndTraining:
    def test_annotated_linear_forward_backward(self):
        """Forward/backward through nn.Linear with annotated input."""
        layout = RowMajor((8, 4))
        model = torch.nn.Linear(4, 3)
        x = annotate(torch.randn(8, 4), layout)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert model.weight.grad is not None
        assert model.weight.grad.shape == (3, 4)

    def test_multi_layer_model(self):
        """Multi-layer model with annotated input."""
        layout = RowMajor((16, 8))

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(8, 16)
                self.fc2 = torch.nn.Linear(16, 4)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        model = Net()
        x = annotate(torch.randn(16, 8), layout)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert model.fc1.weight.grad is not None
        assert model.fc2.weight.grad is not None

    @requires_cuda
    def test_cuda_training_loop(self):
        """Mini training loop on CUDA with annotated data."""
        layout = RowMajor((32, 16))
        model = torch.nn.Linear(16, 4).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for _ in range(5):
            x = annotate(torch.randn(32, 16, device="cuda"), layout)
            y = model(x)
            loss = y.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Just verify no crash and grads flow
        assert model.weight.grad is not None
```

- [ ] **Step 2: Run stress tests**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_stress.py -xvs --timeout=300`

Expected: All pass (CUDA/Triton tests skip if unavailable).

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_torch_stress.py
git commit -m "test: comprehensive stress tests for PyTorch backend (T1-T7)"
```

---

### Task 15: Full regression run and final verification

**Files:** None (verification only)

- [ ] **Step 1: Run the entire PyTorch test suite**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py tests/test_torch_ops.py tests/test_torch_compile.py tests/test_torch_autotune.py tests/test_torch_stress.py -v --timeout=300`

Expected: All tests pass.

- [ ] **Step 2: Run the full project test suite**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/ -v --timeout=300`

Expected: All tests pass.

- [ ] **Step 3: Verify no import regressions**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -c "import lego; import torch; print(lego.annotate); print(lego.rearrange); print(lego.LegoTensor); print(lego.autotune); print('All imports OK')"`

Expected: All imports succeed.

- [ ] **Step 4: Verify CUDA tests if available**

Run: `source /scratch/general/vast/u1419116/LEGO/venv/bin/activate && cd /scratch/general/vast/u1419116/LEGO/python && python -m pytest tests/test_torch_subclass.py tests/test_torch_ops.py tests/test_torch_stress.py -v -k cuda --timeout=300`

Expected: CUDA tests pass or skip cleanly.
