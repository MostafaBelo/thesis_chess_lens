from setuptools import setup, Extension, find_packages
import pybind11
import sys

# ext_modules = [
#     Extension(
#         "HMM._hmmcpp",  # this becomes importable as `import HMM._hmmcpp`
#         sources=["src/HMM/cpp/bindings.cpp",
#                  "src/HMM/cpp/ChessGameState.cpp", "src/HMM/cpp/ChessHMM.cpp"],
#         include_dirs=[pybind11.get_include(), "src/HMM/cpp"],
#         language="c++",
#         extra_compile_args=["-std=c++17"],  # or whatever standard you need
#     ),
# ]

# setup(
#     name="ChessHMM",
#     version="0.1",
#     package_dir={"": "src"},
#     packages=find_packages(where="src"),
#     ext_modules=ext_modules,
# )

opencv_include = "/usr/include/opencv4"
opencv_lib = "/usr/lib/x86_64-linux-gnu"

ext_modules = [
    Extension(
        "HMM._hmmcpp",
        sources=[
            "src/HMM/cpp/bindings.cpp",
            "src/HMM/cpp/ChessGameState.cpp",
            "src/HMM/cpp/ChessHMM.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "src/HMM/cpp",
        ],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
    Extension(
        "PieceDetection.PieceCropper_3D._croppercpp",
        sources=[
            "src/PieceDetection/PieceCropper_3D/cpp/bindings.cpp",
            "src/PieceDetection/PieceCropper_3D/cpp/PieceCropper.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "src/PieceDetection/PieceCropper_3D/cpp",
            # ✅ Add this if your header is there
            # "src/PieceDetection/PieceCropper_3D/include",
            opencv_include
        ],
        library_dirs=[
            opencv_lib,  # ✅ add this
        ],
        libraries=[
            "opencv_core", "opencv_imgproc", "opencv_highgui", "opencv_imgcodecs",
        ],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
]

setup(
    name="ChessAndPieceModules",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
)
