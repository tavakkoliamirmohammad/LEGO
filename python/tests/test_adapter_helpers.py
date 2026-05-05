"""Regression test for decorator-chain helpers extracted from CutileAdapter."""

import functools
from lego.frontends._adapter import (
    try_fn_chain_unwrap,
    try_py_func_unwrap,
    try_wrapped_unwrap,
    walk_to_source_fn,
)


def test_fn_chain_unwrap_walks_dot_fn_chain():
    def inner(): pass
    class Wrapper:
        def __init__(self, fn): self.fn = fn
    outer = Wrapper(Wrapper(inner))
    fn, wrappers = try_fn_chain_unwrap(outer)
    assert fn is inner
    assert wrappers == [outer, outer.fn]


def test_fn_chain_unwrap_returns_input_when_no_chain():
    def plain(): pass
    fn, wrappers = try_fn_chain_unwrap(plain)
    assert fn is plain
    assert wrappers == []


def test_py_func_unwrap_extracts_one_level():
    def real(): pass
    class Numba:
        def __init__(self, fn): self.py_func = fn
    n = Numba(real)
    fn, wrappers = try_py_func_unwrap(n)
    assert fn is real
    assert wrappers == [n]


def test_py_func_unwrap_returns_input_when_no_attr():
    def plain(): pass
    fn, wrappers = try_py_func_unwrap(plain)
    assert fn is plain
    assert wrappers == []


def test_wrapped_unwrap_handles_functools():
    def inner(): pass
    @functools.wraps(inner)
    def outer(): pass
    fn, wrappers = try_wrapped_unwrap(outer)
    assert fn is inner
    assert wrappers == [outer]


def test_walk_to_source_fn_follows_src_fn():
    def base(): pass
    class Layer:
        def __init__(self, fn): self.src_fn = fn
    chained = Layer(Layer(base))
    assert walk_to_source_fn(chained) is base


def test_walk_to_source_fn_no_attr():
    def plain(): pass
    assert walk_to_source_fn(plain) is plain
