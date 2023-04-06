#include "base.hpp"
#include "logging.hpp"
#include "pybindi.h"

#include "problems.hpp"
#include <memory>

void pybind_problems(py::module_ &m)
{
    // Evaluation Logging
    py::class_<EvaluationLogger, ObjectiveFunction, std::shared_ptr<EvaluationLogger>>(m, "EvaluationLogger")
        .def(py::init<std::shared_ptr<ObjectiveFunction>, std::shared_ptr<BaseLogger>>(),
             py::arg("wrapped"),
             py::arg("logger"));

    // General purpose functions
    py::class_<DiscreteObjectiveFunction, ObjectiveFunction, std::shared_ptr<DiscreteObjectiveFunction>>(
        m, "DiscreteObjectiveFunction")
        .def(py::init<std::function<double(std::vector<char> &)>, size_t, std::vector<char>>(),
             py::arg("evaluation_function"),
             py::arg("l"),
             py::arg("alphabet_size"));
    py::class_<ContinuousObjectiveFunction, ObjectiveFunction, std::shared_ptr<ContinuousObjectiveFunction>>(
        m, "ContinuousObjectiveFunction")
        .def(py::init<std::function<double(std::vector<double> &)>, size_t>(),
             py::arg("evaluation_function"),
             py::arg("l"));

    // -- Benchmark Functions --

    // Onemax & Zeromax
    py::class_<OneMax, ObjectiveFunction, std::shared_ptr<OneMax>>(m, "OneMax")
        .def(py::init<size_t, size_t>(), py::arg("l"), py::arg("index") = 0);
    py::class_<ZeroMax, ObjectiveFunction, std::shared_ptr<ZeroMax>>(m, "ZeroMax")
        .def(py::init<size_t, size_t>(), py::arg("l"), py::arg("index") = 0);

    // Maxcut
    py::class_<Edge>(m, "Edge").def(py::init<size_t, size_t, double>(), py::arg("i"), py::arg("j"), py::arg("w"));
    py::class_<MaxCutInstance>(m, "MaxCutInstance")
        .def(py::init<size_t, size_t, std::vector<Edge>>(),
             py::arg("num_vertices"),
             py::arg("num_edges"),
             py::arg("edges"));
    py::class_<MaxCut, ObjectiveFunction, std::shared_ptr<MaxCut>>(m, "MaxCut")
        .def(py::init<MaxCutInstance, size_t>(), py::arg("instance"), py::arg("index") = 0)
        .def(py::init<std::filesystem::path &, size_t>(), py::arg("path"), py::arg("index") = 0);

    // NK-Landscape
    py::class_<NKLandscape, ObjectiveFunction, std::shared_ptr<NKLandscape>>(m, "NKLandscape")
        .def(py::init<std::filesystem::path &, size_t>(), py::arg("path"), py::arg("index") = 0);

    // Best-of-Traps
    py::class_<ConcatenatedPermutedTrap>(m, "ConcatenatedPermutedTrap")
        .def(py::init<size_t, size_t, std::vector<size_t>, std::vector<char>>(),
             py::arg("number_of_parameters"),
             py::arg("block_size"),
             py::arg("permutation"),
             py::arg("optimum"));
    py::class_<BestOfTrapsInstance>(m, "BestOfTrapsInstance")
        .def(py::init<size_t, std::vector<ConcatenatedPermutedTrap>>(),
             py::arg("l"),
             py::arg("concatenatedPermutedTraps"));

    py::class_<BestOfTraps, ObjectiveFunction, std::shared_ptr<BestOfTraps>>(m, "BestOfTraps")
        .def(py::init<BestOfTrapsInstance, size_t>(), py::arg("instance"), py::arg("index") = 0)
        .def(py::init<std::filesystem::path &, size_t>(), py::arg("path"), py::arg("index") = 0);

    py::class_<WorstOfTraps, ObjectiveFunction, std::shared_ptr<WorstOfTraps>>(m, "WorstOfTraps")
        .def(py::init<BestOfTrapsInstance, size_t>(), py::arg("instance"), py::arg("index") = 0)
        .def(py::init<std::filesystem::path &, size_t>(), py::arg("path"), py::arg("index") = 0);

    // Compose multiple functions together.
    py::class_<Compose, ObjectiveFunction, std::shared_ptr<Compose>>(m, "Compose")
        .def(py::init<std::vector<std::shared_ptr<ObjectiveFunction>>>(), py::arg("objective_functions"));

    //
    py::class_<HierarchicalDeceptiveTrap, ObjectiveFunction, std::shared_ptr<HierarchicalDeceptiveTrap>>(
        m, "HierarchicalDeceptiveTrap")
        .def(py::init<size_t, size_t, size_t>(), py::arg("l"), py::arg("k") = 3, py::arg("index") = 0);

    // Exceptions
    py::register_exception<missing_file>(m, "FileNotFound");
    py::register_exception<invalid_instance>(m, "InvalidInstance");
}