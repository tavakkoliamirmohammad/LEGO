"""DSLAdapter — abstract interface for DSL-specific hooks in the LEGO rewriter."""

import ast
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Set, Tuple

from lego.python_printer import LEGOPythonCodePrinter


class DSLAdapter(ABC):
    @abstractmethod
    def unwrap(self, fn) -> Tuple[Callable, Callable, List[Any]]:
        """Strip DSL decorators -> (source_fn, original_fn, wrappers).

        source_fn:    the raw function whose source can be inspected
        original_fn:  the innermost callable (may == source_fn)
        wrappers:     list of decorator objects to re-apply after rewriting
        """

    @abstractmethod
    def find_runtime_vars(self, func_def: ast.FunctionDef) -> Set[str]:
        """Return variable names that must NOT be evaluated at decoration time."""

    @abstractmethod
    def get_code_printer(self) -> LEGOPythonCodePrinter:
        """Return a SymPy code printer with DSL-specific rendering."""

    @abstractmethod
    def compile_and_wrap(self, new_source: str, tree: ast.Module,
                         original_fn: Callable, wrappers: List[Any],
                         return_source: bool) -> Any:
        """Compile transformed source, re-apply DSL decorators, return result."""

    def get_rewriter_options(self) -> dict:
        """Return DSL-specific options for the rewriter. Default: empty."""
        return {}

    def try_block_ptr_pattern(self, stmt, eval_env, printer):
        """Try to match a block-ptr pattern in the AST. Default: no match."""
        return None


class SourceGenAdapter(DSLAdapter):
    """Concrete adapter for source-to-source code generation.

    All source-gen frontends (C++, CUDA C, Fortran, GLSL, JS, Julia, Rust)
    share identical behaviour: no decorator unwrapping, no runtime vars,
    and compile_and_wrap just returns the source text.  Only the code
    printer differs.
    """

    def __init__(self, printer):
        self._printer = printer

    def unwrap(self, fn):
        return fn, fn, []

    def find_runtime_vars(self, func_def):
        return set()

    def get_code_printer(self):
        return self._printer

    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        return new_source


def source_generate(fn, printer, **kwargs):
    """Generate source code from a function with LEGO layout expressions."""
    from lego.rewriter import rewrite
    return rewrite(fn, SourceGenAdapter(printer), return_source=True, **kwargs)


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
    exec(code_obj, namespace)  # noqa: S102 — intentional dynamic exec

    if not _save:
        atexit.register(
            lambda f=temp_file: os.remove(f) if os.path.exists(f) else None)

    transformed_fn = namespace[original_fn.__name__]
    transformed_fn.__code__ = transformed_fn.__code__.replace(
        co_filename=temp_file)

    return namespace, transformed_fn
