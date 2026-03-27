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
