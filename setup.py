"""Installation script for the 'wbc_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.8.1",
    "warp-lang==1.12.0",
    "scipy==1.17.0",
    "swanlab>=0.8,<0.9",
]

# Installation operation
setup(
    name="wbc_mjlab",
    packages=["src"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
