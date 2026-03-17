"""Dynamic setup config for wheel builds.

pyproject.toml handles metadata. This file only provides:
- package_data discovery (shared libs under mlir/ and lego/)
- has_ext_modules (forces platform-tagged wheel)
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
    packages=packages,
    package_data=package_data,
    has_ext_modules=lambda: True,
)
