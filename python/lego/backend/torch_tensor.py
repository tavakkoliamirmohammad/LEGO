"""
LEGO PyTorch Tensor Subclass

Provides LegoTensor, a torch.Tensor subclass that carries layout metadata.
Storage holds data in physical (transformed) order.

Layout-aware dispatch: when two LegoTensors share the same layout,
elementwise ops operate directly on physical storage (zero-copy).
Mismatched layouts fall back to logical order.
"""

import warnings

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    import numpy as np
    from lego.backend.compiler import LayoutCompiler

    # ====================================================================
    # Layout-aware op dispatch table
    # ====================================================================

    _LAYOUT_AWARE_OPS = {}  # torch_func -> handler
    _WARNED_OPS = set()     # avoid spamming warnings

    def _register_layout_op(torch_func):
        """Decorator to register a layout-aware handler for a torch function."""
        def decorator(handler):
            _LAYOUT_AWARE_OPS[torch_func] = handler
            return handler
        return decorator

    def _same_layout(tensors):
        """Check if all LegoTensors in the list share the same layout object."""
        layouts = []
        for t in tensors:
            if isinstance(t, LegoTensor) and t._lego_layout is not None:
                layouts.append(t._lego_layout)
        if len(layouts) < 2:
            return len(layouts) == 1
        first = layouts[0]
        return all(l is first or l == first for l in layouts[1:])

    def _extract_lego_tensors(args):
        """Extract all LegoTensors from nested args."""
        result = []
        if isinstance(args, (list, tuple)):
            for a in args:
                result.extend(_extract_lego_tensors(a))
        elif isinstance(args, LegoTensor):
            result.append(args)
        return result

    def _to_physical(t):
        """Get physical data as plain torch.Tensor."""
        if isinstance(t, LegoTensor):
            return torch.Tensor._make_subclass(torch.Tensor, t)
        return t

    def _wrap_result(result, layout, fwd_perm, inv_perm):
        """Wrap a plain tensor result as a LegoTensor with the same layout."""
        if isinstance(result, torch.Tensor) and not isinstance(result, LegoTensor):
            return LegoTensor(result, layout, fwd_perm, inv_perm)
        return result

    # ====================================================================
    # Elementwise ops — operate on physical storage when layouts match
    # ====================================================================

    def _make_elementwise_handler(torch_func):
        """Create a layout-aware handler for a binary/unary elementwise op."""
        def handler(*args, **kwargs):
            lego_tensors = _extract_lego_tensors(args)
            if not lego_tensors:
                return torch_func(*args, **kwargs)

            # Check if all LegoTensors share the same layout
            if _same_layout(lego_tensors):
                ref = lego_tensors[0]
                layout = ref._lego_layout
                fwd = ref._fwd_perm
                inv = ref._inv_perm

                # Convert args to physical tensors
                new_args = []
                for a in args:
                    if isinstance(a, (list, tuple)):
                        new_args.append(type(a)(_to_physical(x) for x in a))
                    else:
                        new_args.append(_to_physical(a))

                result = torch_func(*new_args, **kwargs)
                return _wrap_result(result, layout, fwd, inv)
            else:
                # Mismatched layouts — fall back to logical
                new_args = []
                for a in args:
                    if isinstance(a, LegoTensor):
                        new_args.append(a.to_logical())
                    else:
                        new_args.append(a)
                return torch_func(*new_args, **kwargs)
        return handler

    # Register elementwise ops
    _ELEMENTWISE_OPS = [
        torch.add, torch.sub, torch.mul, torch.div,
        torch.neg, torch.abs, torch.exp, torch.log,
        torch.sigmoid, torch.tanh, torch.relu,
        torch.sqrt, torch.rsqrt, torch.reciprocal,
        torch.sin, torch.cos,
        torch.Tensor.add, torch.Tensor.sub, torch.Tensor.mul, torch.Tensor.div,
        torch.Tensor.neg, torch.Tensor.abs, torch.Tensor.exp, torch.Tensor.log,
        torch.Tensor.sigmoid, torch.Tensor.tanh, torch.Tensor.relu,
    ]

    for _op in _ELEMENTWISE_OPS:
        _LAYOUT_AWARE_OPS[_op] = _make_elementwise_handler(_op)

    # ====================================================================
    # Reduction ops — full reductions are order-independent
    # ====================================================================

    def _make_full_reduction_handler(torch_func):
        """For reductions with no dim arg, operate on physical data directly."""
        def handler(*args, **kwargs):
            lego_tensors = _extract_lego_tensors(args)
            if not lego_tensors:
                return torch_func(*args, **kwargs)

            # Check if this is a full reduction (no dim specified)
            dim = kwargs.get('dim', None)
            # Also check positional dim arg for functions like torch.sum(input, dim)
            if dim is None and len(args) >= 2 and isinstance(args[1], (int, tuple, list)):
                dim = args[1]

            if dim is None:
                # Full reduction — order-independent, use physical data
                new_args = [_to_physical(a) if isinstance(a, LegoTensor) else a for a in args]
                return torch_func(*new_args, **kwargs)
            else:
                # Axis-specific — fall back to logical order
                new_args = [a.to_logical() if isinstance(a, LegoTensor) else a for a in args]
                return torch_func(*new_args, **kwargs)
        return handler

    _REDUCTION_OPS = [
        torch.sum, torch.mean, torch.prod,
        torch.max, torch.min, torch.norm,
        torch.Tensor.sum, torch.Tensor.mean, torch.Tensor.prod,
    ]

    for _op in _REDUCTION_OPS:
        _LAYOUT_AWARE_OPS[_op] = _make_full_reduction_handler(_op)

    # ====================================================================
    # Layout-aware matmul
    # ====================================================================

    def _matmul_handler(*args, **kwargs):
        """Layout-aware matmul. Falls back to logical for non-standard layouts."""
        new_args = [a.to_logical() if isinstance(a, LegoTensor) else a for a in args]
        return torch.matmul(*new_args, **kwargs)

    _LAYOUT_AWARE_OPS[torch.matmul] = _matmul_handler
    _LAYOUT_AWARE_OPS[torch.mm] = _matmul_handler
    _LAYOUT_AWARE_OPS[torch.Tensor.matmul] = _matmul_handler
    _LAYOUT_AWARE_OPS[torch.Tensor.mm] = _matmul_handler

    # Also handle @ operator
    _LAYOUT_AWARE_OPS[torch.Tensor.__matmul__] = _matmul_handler

    # ====================================================================
    # LegoTensor class
    # ====================================================================

    class LegoTensor(torch.Tensor):
        """torch.Tensor subclass carrying LEGO layout metadata.

        Storage holds data in physical (transformed) order.
        Provides to_logical() / to_physical() for explicit conversion.

        When two LegoTensors share the same layout, elementwise ops
        (add, mul, relu, etc.) operate directly on physical storage
        without permutation overhead.
        """

        @staticmethod
        def __new__(cls, data, layout, fwd_perm=None, inv_perm=None):
            return torch.Tensor._make_subclass(cls, data)

        def __init__(self, data, layout, fwd_perm=None, inv_perm=None):
            self._lego_layout = layout
            self._fwd_perm = fwd_perm
            self._inv_perm = inv_perm

        def to_logical(self):
            """Convert to logical (row-major) order as a regular torch.Tensor."""
            plain = torch.Tensor._make_subclass(torch.Tensor, self)
            if self._inv_perm is not None:
                flat = plain.contiguous().view(-1)
                return flat[self._inv_perm].view(plain.shape)
            return self._lego_layout.inverse_transform(plain)

        def to_physical(self):
            """Return underlying physical-order data as regular torch.Tensor."""
            return torch.Tensor._make_subclass(torch.Tensor, self)

        @property
        def layout_info(self):
            return self._lego_layout

        def __repr__(self):
            plain = torch.Tensor._make_subclass(torch.Tensor, self)
            return f"LegoTensor(layout={self._lego_layout!r})\n{plain!r}"

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            """Layout-aware dispatch for PyTorch operations.

            If the op is in the layout-aware dispatch table and all LegoTensors
            share the same layout, the op executes on physical storage directly.
            Otherwise, falls back to converting to logical order.
            """
            kwargs = kwargs or {}

            if func in _LAYOUT_AWARE_OPS:
                return _LAYOUT_AWARE_OPS[func](*args, **kwargs)

            # Fallback: convert to logical order
            if func not in _WARNED_OPS:
                _WARNED_OPS.add(func)
                func_name = getattr(func, '__name__', str(func))
                warnings.warn(
                    f"{func_name} not layout-aware, converting to logical order",
                    stacklevel=2,
                )

            new_args = []
            for a in args:
                if isinstance(a, LegoTensor):
                    new_args.append(a.to_logical())
                else:
                    new_args.append(a)
            return func(*new_args, **kwargs)

    def as_lego_tensor(tensor, layout):
        """Transform a tensor and wrap as LegoTensor."""
        transformed = layout.transform(tensor)

        compiler = LayoutCompiler(layout._layout, layout._shape, "i64")
        fwd, inv = compiler.get_permutation_table()
        fwd_t = torch.from_numpy(fwd).to(tensor.device)
        inv_t = torch.from_numpy(inv).to(tensor.device)

        return LegoTensor(transformed, layout, fwd_t, inv_t)

else:
    LegoTensor = None
    as_lego_tensor = None
