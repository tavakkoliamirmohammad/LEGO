#!/usr/bin/env python3
"""Simple static file server for the LEGO Layout Visualizer.

The compiler runs entirely in the browser via WebAssembly — no Python
compilation backend is needed.  This server just serves static files
with the correct MIME types for .wasm and .js.

Usage:
    python viz/serve.py [--port PORT]
    # Then open http://localhost:8080
"""

import argparse
import http.server
import functools
import os

class WasmHandler(http.server.SimpleHTTPRequestHandler):
    """Adds WASM MIME type and CORS headers for local development."""

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.wasm': 'application/wasm',
        '.mjs': 'application/javascript',
    }

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description='LEGO Viz static server')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    viz_dir = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(WasmHandler, directory=viz_dir)

    with http.server.HTTPServer(('', args.port), handler) as httpd:
        print(f'Serving LEGO Visualizer at http://localhost:{args.port}')
        print(f'  Static files: {viz_dir}')
        print(f'  WASM compiler: wasm/lego_driver.wasm')
        print('Press Ctrl+C to stop.')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nShutting down.')


if __name__ == '__main__':
    main()
