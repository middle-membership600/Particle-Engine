#include <pybind11/pybind11.h>
#include <pybind11/embed.h> // for Python embedding

namespace py = pybind11;

PYBIND11_MODULE(pinn_cpp_wrapper, m) {
    m.def("predict", [](float x) {
        py::gil_scoped_acquire acquire; // Acquire GIL
        py::module_ pinn_model = py::module_::import("PINN.inference");
        py::object result = pinn_model.attr("predict")(x);
        py::gil_scoped_release release; // Release GIL
        return result.cast<float>(); // Assuming the result is a float or a list of floats
    });
}
