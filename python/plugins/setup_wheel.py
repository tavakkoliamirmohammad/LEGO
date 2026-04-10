"""Dynamic setup config for LEGO plugin wheel builds.

pyproject.toml handles metadata. This file only provides:
- package_data discovery (shared libs under _mlir_libs/)
- has_ext_modules (forces platform-tagged wheel)
"""
import os
from setuptools import setup, find_packages

packages = find_packages()

package_data = {}
for pkg in packages:
    pkg_dir = pkg.replace(".", os.sep)
    if os.path.isdir(pkg_dir):
        for root, _dirs, files in os.walk(pkg_dir):
            rel_pkg = root.replace(os.sep, ".")
            data = [f for f in files if not f.endswith((".py", ".pyc"))]
            if data:
                package_data[rel_pkg] = data

setup(
    packages=packages,
    package_data=package_data,
    has_ext_modules=lambda: True,
)
