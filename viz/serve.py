#!/usr/bin/env python3
"""
LEGO Layout Visualizer — local development server.

Endpoints:
  GET  /           — serves the static visualization app
  POST /compile    — compiles a layout to WASM or returns JSON mapping

Usage:
  python viz/serve.py                    # default port 8080
  python viz/serve.py --port 3000        # custom port

Requires: PYTHONPATH set to include python/ (for lego imports).
"""

import argparse
import json
import os
import sys
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

# Add the LEGO build python packages to path (has compiled MLIR bindings).
# Then sync any new source modules (like wasm.py) from the source tree
# into the build tree so they're importable alongside the MLIR bindings.
LEGO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_build_pkg = os.path.join(LEGO_ROOT, "build", "python_packages", "lego")
_src_pkg = os.path.join(LEGO_ROOT, "python")

if os.path.isdir(_build_pkg):
    # Sync new/changed source files into build tree
    import shutil
    _src_wasm = os.path.join(_src_pkg, "lego", "backend", "wasm.py")
    _dst_wasm = os.path.join(_build_pkg, "lego", "backend", "wasm.py")
    if os.path.exists(_src_wasm):
        shutil.copy2(_src_wasm, _dst_wasm)
    sys.path.insert(0, _build_pkg)
else:
    sys.path.insert(0, _src_pkg)


def _compile_layout(code, shape, extra_dims=None):
    """Execute user layout code and compile to mapping JSON.

    Returns a dict with {mapping, shape, total} ready for JSON serialization.
    """
    import sympy as sp
    from lego.core import OrderBy, Row, Col, RegP, GenP, GroupBy, TileByLayout

    # Build namespace with LEGO primitives and dimension values
    namespace = {
        "OrderBy": OrderBy,
        "Row": Row,
        "Col": Col,
        "RegP": RegP,
        "GenP": GenP,
        "GroupBy": GroupBy,
        "sp": sp,
    }

    # Inject dimension values as plain integers
    dim_names = ["M", "N", "K", "P", "Q"]
    for i, s in enumerate(shape):
        if i < len(dim_names):
            namespace[dim_names[i]] = int(s)

    # Inject extra dims (BM, BN, etc.)
    if extra_dims:
        namespace.update({k: int(v) for k, v in extra_dims.items()})

    # Execute user code
    exec(code, namespace)

    layout = namespace.get("L")
    if layout is None:
        raise ValueError(
            "Layout not found. Your code must assign to a variable named 'L'.\n"
            "Example: L = OrderBy(Row(M, N)).GroupBy([(M, N)])"
        )

    # Try WASM compilation first, fall back to JIT-based JSON mapping
    try:
        from lego.backend.wasm import compile_to_wasm
        wasm_bytes = compile_to_wasm(layout, shape)
        return {"type": "wasm", "data": wasm_bytes}
    except Exception:
        pass

    # Fallback: compute mapping via MLIR JIT
    try:
        from lego.backend.wasm import compile_mapping_json
        result = compile_mapping_json(layout, shape)
        result["type"] = "json"
        return result
    except Exception as e:
        # Try symbolic fallback, but preserve the real error
        jit_error = str(e)
        try:
            return _eval_symbolic(layout, shape)
        except Exception:
            # Surface the original JIT error with helpful context
            raise ValueError(
                f"Layout compilation failed: {jit_error}\n\n"
                f"TileBy syntax: L = OrderBy(Row(M, N)).TileBy([M//BM, N//BN], [BM, BN])\n"
                f"GroupBy syntax: L = OrderBy(Row(M, N)).GroupBy([(M, N)])"
            ) from None


def _eval_symbolic(layout, shape):
    """Pure symbolic evaluation fallback — no MLIR needed for simple layouts."""
    import sympy as sp

    mapping = []
    total = 1
    for s in shape:
        total *= int(s)

    if len(shape) == 2:
        M, N = int(shape[0]), int(shape[1])
        syms = sp.symbols("i j", integer=True)
        try:
            expr = layout.apply(*syms)
            for i in range(M):
                for j in range(N):
                    flat = int(expr.subs({syms[0]: i, syms[1]: j}))
                    mapping.append([i, j, flat])
            return {"type": "json", "mapping": mapping, "shape": list(shape), "total": total}
        except Exception:
            pass

    raise ValueError("Could not evaluate layout. Ensure MLIR is built.")


class VizHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves static files and the /compile API."""

    def __init__(self, *args, **kwargs):
        # Serve files from the viz/ directory
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/compile":
            self._handle_compile()
        else:
            self.send_error(404, "Not Found")

    def _handle_compile(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            code = body["code"]
            shape = tuple(body["shape"])
            extra_dims = body.get("extra_dims", {})

            result = _compile_layout(code, shape, extra_dims)

            if result["type"] == "wasm":
                self.send_response(200)
                self.send_header("Content-Type", "application/wasm")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(result["data"])
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc(),
            }).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Quieter logging — skip static file requests."""
        if "POST" in str(args):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="LEGO Layout Visualizer")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), VizHandler)
    print(f"LEGO Visualizer: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
