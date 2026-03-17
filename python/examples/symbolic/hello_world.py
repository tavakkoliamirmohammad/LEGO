"""Symbolic LEGO hello world — CPU only, no MLIR build required."""
import sympy as sp
from lego.frontends.symbolic import OrderBy, Row, GroupBy

M, N = sp.symbols('M N', integer=True, positive=True)
i, j = sp.symbols('i j', integer=True, positive=True)
L = OrderBy(Row(M, N)).GroupBy([(M, N)])
print(f"Forward:  L[i, j] = {L[i, j]}")        # i*N + j
x = sp.Symbol('x', integer=True, positive=True)
print(f"Inverse:  L.inv(x) = {L.inv(x)}")       # [floor(x/N), Mod(x, N)]
