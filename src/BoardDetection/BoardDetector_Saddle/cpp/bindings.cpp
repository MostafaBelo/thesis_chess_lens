#include <pybind11/pybind11.h>
#include "BoardSaddle.h"

namespace py = pybind11;

PYBIND11_MODULE(_boardsaddlecpp, m) {
    m.doc() = "Board Detector (Saddle) C++ core bindings";

    py::class_<ChessboardDetectionConfig>(m, "DetectionConfig")
        .def(py::init<>())
        .def_readwrite("min_pts_needed", &ChessboardDetectionConfig::min_pts_needed)
        .def_readwrite("max_pts_needed", &ChessboardDetectionConfig::max_pts_needed)
        .def_readwrite("max_px_dist", &ChessboardDetectionConfig::max_px_dist);
    
    m.def("detect_corners", &detectChessboardCorners,
          py::arg("image"),
          py::arg("config") = ChessboardDetectionConfig(),
          "Detect chessboard corners in an image");
}