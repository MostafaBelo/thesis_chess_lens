#include <pybind11/pybind11.h>
#include "ChessGameState.h"
#include "ChessHMM.h"

namespace py = pybind11;

PYBIND11_MODULE(_hmmcpp, m) {
    m.doc() = "Chess HMM C++ core bindings";

    pybind11::class_<ChessMove>(m, "ChessMove")
        .def(pybind11::init<string, string>())
        .def("__str__", &ChessMove::str)
        .def("__repr__", &ChessMove::repr);

    pybind11::class_<ChessGameState>(m, "ChessGameState")
        .def(py::init<string>(), py::arg("fen") = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        .def("__str__", &ChessGameState::str)
        .def("__repr__", &ChessGameState::repr)

        .def("eval_next", &ChessGameState::eval_next)
        // .def("eval_prob", &ChessGameState::eval_prob)

        .def("get_fen", &ChessGameState::get_fen)
        ;

    pybind11::class_<GameStateFactory>(m, "GameStateFactory")
        .def(pybind11::init<>())
        .def("get_state", &GameStateFactory::get_state);

    pybind11::class_<ChessHMM>(m, "ChessHMM")
        .def(pybind11::init<int, string>(), py::arg("max_width"), py::arg("fen") = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

        .def("top_t", &ChessHMM::top_t)
        .def("top_bind_t", &ChessHMM::top_bind_t)

        .def("set_probs", &ChessHMM::set_probs, py::arg("timestep"), py::arg("obs_probs"))
        .def("bind", &ChessHMM::bind, py::arg("timestep"))
        
        .def("print", &ChessHMM::print, py::arg("timestep"))
        .def("get_history", &ChessHMM::get_history, py::arg("include_non_bound") = false)
        .def("get_pgn", &ChessHMM::get_pgn);
}
