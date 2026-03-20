"""
LEGO PyTorch Tensor Subclass

Provides LegoTensor, a torch.Tensor subclass that carries layout metadata.
Storage holds data in physical (transformed) order.
"""

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    from lego.backend.compiler import LayoutCompiler

    class LegoTensor(torch.Tensor):
        """torch.Tensor subclass carrying LEGO layout metadata.

        Storage holds data in physical (transformed) order.
        Provides to_logical() / to_physical() for explicit conversion.
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
            """For most ops, convert to logical order first."""
            kwargs = kwargs or {}
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
