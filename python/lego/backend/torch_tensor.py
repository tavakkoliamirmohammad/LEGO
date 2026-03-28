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

    # ====================================================================
    # Name-based op classification
    # ====================================================================

    _ELEMENTWISE_NAMES = frozenset({
        'add', 'sub', 'mul', 'div', 'neg', 'abs', 'exp', 'log',
        'sigmoid', 'tanh', 'relu', 'sqrt', 'rsqrt', 'reciprocal',
        'sin', 'cos', 'pow', 'clamp', 'where',
        'float', 'half', 'double', 'int',
        '__add__', '__radd__', '__sub__', '__rsub__',
        '__mul__', '__rmul__', '__truediv__', '__rtruediv__',
        '__neg__', '__abs__',
    })

    _FULL_REDUCTION_NAMES = frozenset({
        'sum', 'mean', 'prod', 'max', 'min', 'norm',
        'any', 'all',
    })

    _warned_fallback_ops = set()

    def _classify_op(func):
        """Classify a torch function by its short name."""
        name = getattr(func, '__name__', '')
        if name in _ELEMENTWISE_NAMES:
            return 'elementwise'
        if name in _FULL_REDUCTION_NAMES:
            return 'reduction'
        return None

    def _same_layout(tensors):
        """Check if all LegoTensors in the list share the same layout."""
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

    # ====================================================================
    # LegoTensor
    # ====================================================================

    class LegoTensor(torch.Tensor):
        """torch.Tensor subclass carrying LEGO layout metadata.

        Storage holds data in physical (transformed) order.
        The LegoLayout is the single source of truth for permutation state.
        """

        @staticmethod
        def __new__(cls, data, layout):
            return torch.Tensor._make_subclass(cls, data)

        def __init__(self, data, layout):
            self._lego_layout = layout

        def to_logical(self):
            """Convert to logical (row-major) order as a regular torch.Tensor."""
            plain = torch.Tensor._make_subclass(torch.Tensor, self)
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

            Uses name-based classification to determine op behavior:
            - Elementwise ops on same-layout tensors operate on physical storage
            - Full reductions (no dim) operate on physical storage
            - Everything else falls back to logical order
            """
            kwargs = kwargs or {}
            lego_tensors = _extract_lego_tensors(args)

            if not lego_tensors:
                new_args = [_to_physical(a) if isinstance(a, LegoTensor) else a for a in args]
                return func(*new_args, **kwargs)

            category = _classify_op(func)
            ref = lego_tensors[0]

            # Elementwise: preserve layout when all inputs share it
            if category == 'elementwise' and _same_layout(lego_tensors):
                layout = ref._lego_layout
                new_args = []
                for a in args:
                    if isinstance(a, (list, tuple)):
                        new_args.append(type(a)(_to_physical(x) for x in a))
                    else:
                        new_args.append(_to_physical(a))
                result = func(*new_args, **kwargs)
                if isinstance(result, torch.Tensor) and not isinstance(result, LegoTensor):
                    return LegoTensor(result, layout)
                return result

            # Full reductions (no dim): order-independent
            if category == 'reduction':
                dim = kwargs.get('dim', None)
                if dim is None and len(args) >= 2 and isinstance(args[1], (int, tuple, list)):
                    dim = args[1]
                if dim is None:
                    new_args = [_to_physical(a) if isinstance(a, LegoTensor) else a for a in args]
                    return func(*new_args, **kwargs)

            # Fallback: convert to logical order
            func_name = getattr(func, '__name__', str(func))
            if func_name not in _warned_fallback_ops:
                _warned_fallback_ops.add(func_name)
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
        """Transform a tensor and wrap as LegoTensor.

        The LegoLayout owns all permutation/index state. LegoTensor is a thin
        metadata wrapper around the physical-order data.
        """
        transformed = layout.transform(tensor)
        return LegoTensor(transformed, layout)

else:
    LegoTensor = None
    as_lego_tensor = None
