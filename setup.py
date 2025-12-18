from setuptools import setup, find_packages, find_namespace_packages
import os

# Read requirements
with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name="lego_dsl",
    version="1.0.0",
    description="LEGO: A Layout Expression Language for Code Generation of Hierarchical Mapping",
    packages=find_namespace_packages(include=["lego", "lego.*", "mlir", "mlir.*"]),
    include_package_data=True,
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)
