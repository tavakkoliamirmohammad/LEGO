"""LayoutCompiler.compile() must accept a pipeline_name= override."""

import pytest
import inspect
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lego.backend.compiler import LayoutCompiler
from lego.frontends.python_mlir import RowMajor


def test_compile_accepts_pipeline_name_kwarg():
    """The compile() method must accept pipeline_name without TypeError."""
    layout = RowMajor((4, 8))
    compiler = LayoutCompiler(layout._layout, layout._shape, "f32")
    sig = compiler.compile.__code__.co_varnames
    assert 'pipeline_name' in sig, \
        f"compile() must accept pipeline_name kwarg; got {sig}"


def test_compile_default_pipeline_unchanged():
    """Calling compile() without pipeline_name uses 'lego-to-llvm'."""
    sig = inspect.signature(LayoutCompiler.compile)
    assert sig.parameters['pipeline_name'].default == 'lego-to-llvm'
