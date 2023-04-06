#include "pybindi.h"

#include "base.hpp"

void pybind_base(py::module_ &m)
{
    // General wrapper for a datatype
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<DataType>(m, "DataType");
    // Datatypes in the current code
    py::class_<Objective>(m, "Objective", py::buffer_protocol()) //
        .def_readwrite("objectives", &Objective::objectives)
        .def_buffer([](Objective &m) -> py::buffer_info {
            return py::buffer_info(m.objectives.data(),
                                   sizeof(double),
                                   py::format_descriptor<double>::format(),
                                   1,
                                   {m.objectives.size()},
                                   {sizeof(double)});
        });
    m.attr("OBJECTIVE") = OBJECTIVE;
    py::class_<GenotypeCategorical>(m, "GenotypeCategorical", py::buffer_protocol()) //
        .def_readwrite("genotype", &GenotypeCategorical::genotype)
        .def_buffer([](GenotypeCategorical &m) -> py::buffer_info {
            return py::buffer_info(m.genotype.data(),
                                   sizeof(char),
                                   py::format_descriptor<char>::format(),
                                   1,
                                   {m.genotype.size()},
                                   {sizeof(char)});
        });
    m.attr("GENOTYPECATEGORICAL") = GENOTYPECATEGORICAL;

    // TODO: Buffer
    py::class_<GenotypeContinuous>(m, "GenotypeContinuous").def_readwrite("genotype", &GenotypeContinuous::genotype);
    m.attr("GENOTYPECONTINUOUS") = GENOTYPECONTINUOUS;

    py::class_<Population>(m, "Population")
        .def(py::init<>())
        .def("getData", &Population::getDataPython, py::return_value_policy::reference);

    py::class_<Individual>(m, "Individual").def_readonly("i", &Individual::i);
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<IDataUser, std::shared_ptr<IDataUser>>(m, "IDataUser");
    // - objective functions
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<ObjectiveFunction, IDataUser, std::shared_ptr<ObjectiveFunction>>(m, "ObjectiveFunction");

    //   (related utilities & wrappers)
    py::class_<Limiter, ObjectiveFunction, std::shared_ptr<Limiter>>(m, "Limiter")
        .def(py::init<std::shared_ptr<ObjectiveFunction>,
                      std::optional<long long>,
                      std::optional<std::chrono::duration<double>>>(),
             py::arg("wrapping"),
             py::arg("evaluation_limit") = std::nullopt,
             py::arg("time_limit") = std::nullopt)
        .def("restart", &Limiter::restart)
        .def("get_time_spent_ms", &Limiter::get_time_spent_ms)
        .def("get_num_evaluations", &Limiter::get_num_evaluations);
    py::class_<ElitistMonitor, ObjectiveFunction, std::shared_ptr<ElitistMonitor>>(m, "ElitistMonitor")
        .def(py::init<std::shared_ptr<ObjectiveFunction>, std::shared_ptr<IPerformanceCriterion>>(),
             py::arg("wrapping"),
             py::arg("criterion"))
        .def("getElitist", &ElitistMonitor::getElitist);

    py::class_<ObjectiveValuesToReachDetector, ObjectiveFunction, std::shared_ptr<ObjectiveValuesToReachDetector>>(m, "ObjectiveValuesToReachDetector")
        .def(py::init<std::shared_ptr<ObjectiveFunction>, std::vector<std::vector<double>>, bool>(),
             py::arg("wrapping"),
             py::arg("vtrs"),
             py::arg("allow_dominating") = false);
    
    py::register_exception<limit_reached>(m, "LimitReached");
    py::register_exception<evaluation_limit_reached>(m, "EvaluationLimit");
    py::register_exception<time_limit_reached>(m, "TimeLimit");
    py::register_exception<vtr_reached>(m, "AllVTRFound");
}