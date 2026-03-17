"""Wheel build script for LEGO.

Used by CI to create wheels from the build tree.
Expects to be run from build/python_packages/lego/.
"""
import os
from setuptools import setup, find_namespace_packages

packages = find_namespace_packages(include=["mlir", "mlir.*", "lego", "lego.*"])

package_data = {}
for top in ("mlir", "lego"):
    for root, dirs, files in os.walk(top):
        pkg = root.replace(os.sep, ".")
        data = [f for f in files if not f.endswith((".py", ".pyc"))]
        if data:
            package_data[pkg] = data

setup(
    name="lego-layout",
    version="0.1.3",
    description="LEGO: Layout Expression Language for Code Generation",
    python_requires=">=3.12",
    install_requires=["sympy", "numpy"],
    packages=packages,
    package_data=package_data,
    has_ext_modules=lambda: True,
)
