#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;

#ifndef PIECECROPPER
#define PIECECROPPER

py::array_t<uint8_t, py::array::c_style | py::array::forcecast> extract_warped_squares(
    py::array_t<float, py::array::c_style | py::array::forcecast> grid_top,
    py::array_t<float, py::array::c_style | py::array::forcecast> grid_bottom,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img_numpy,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> res
);

#endif