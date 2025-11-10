#include <pybind11/pybind11.h>
#include "PieceCropper.h"

namespace py = pybind11;

PYBIND11_MODULE(_croppercpp, m) {
    m.doc() = "Piece Cropper C++ core bindings";

    m.def("extract_warped_squares", &extract_warped_squares,
        py::arg("grid_top"),
        py::arg("grid_bottom"),
        py::arg("img_numpy"),
        py::arg("res")
    );
}
