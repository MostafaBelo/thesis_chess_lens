from setuptools import setup, Extension, find_packages
import pybind11
import sys

ext_modules = [
    Extension(
        "HMM._hmmcpp",  # this becomes importable as `import HMM._hmmcpp`
        sources=["src/HMM/cpp/bindings.cpp",
                 "src/HMM/cpp/ChessGameState.cpp", "src/HMM/cpp/ChessHMM.cpp"],
        include_dirs=[pybind11.get_include(), "src/HMM/cpp"],
        language="c++",
        extra_compile_args=["-std=c++17"],  # or whatever standard you need
    ),
]

setup(
    name="ChessHMM",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
)
