// #include <iostream>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>

// using namespace std;

// class Calculator {
//     public:
//         Calculator(int value) : value(value) {
//         }

//         int add(int x) {
//             return value + x;
//         }

//     private:
//         int value;
// };

// class Testing {
//     private:
//         int my_val;

//     public:
//         Testing() {
//             Testing::my_val = 3;
//         }

//         void set_val(int new_val) {
//             Testing::my_val = new_val;
//         }

//         int get_val() {
//             return Testing::my_val;
//         }
// };

// int add(int a, int b) {
//     return a + b;
// }

// int subtract(int a, int b) {
//     return a - b;
// }

// pybind11::array_t<double> scale_array(pybind11::array_t<double> input_array, double scalar) {
//     // Request buffer information from the input NumPy array
//     auto buf = input_array.request();
    
//     // Check if the array is one-dimensional
//     if (buf.ndim != 1) throw std::runtime_error("Input array must be 1-dimensional");

//     // Access the raw data pointer
//     double *ptr = static_cast<double *>(buf.ptr);

//     // Create a new NumPy array to store the scaled values
//     auto result = pybind11::array_t<double>(buf.size);
//     auto result_buf = result.request();
//     double *result_ptr = static_cast<double *>(result_buf.ptr);

//     // Perform the scaling operation element-wise
//     for (size_t i = 0; i < buf.size; ++i) {
//         result_ptr[i] = ptr[i] * scalar;
//     }

//     return result; // Return the new NumPy array
// }

// PYBIND11_MODULE(example, m) {
//     m.doc() = "pybind11 example module"; // Optional module docstring

//     m.def("add", &add, "A function that adds two numbers",
//           pybind11::arg("a"), pybind11::arg("b"));

//     m.def("subtract", &subtract, "A function that subtracts two numbers",
//           pybind11::arg("a"), pybind11::arg("b"));

//     pybind11::class_<Calculator>(m, "Calculator")
//         .def(pybind11::init<int>())
//         .def("add", &Calculator::add);

//     pybind11::class_<Testing>(m, "Testing")
//         .def(pybind11::init<>())
//         .def("set_val", &Testing::set_val, "set the value",
//             pybind11::arg("new_val"))
//         .def("get_val", &Testing::get_val);

//     m.def("scale_array", &scale_array, "Scale a NumPy array by a scalar",
//           pybind11::arg("input_array"), pybind11::arg("scalar"));
// }