#include "cppassert.h"

#include "python/pybindi.h"
#include "archive.hpp"

// -- Bindings for certain modules --

void pybind_utilities(py::module_ &m); // python/utilities.cpp
void pybind_base(py::module_ &m); // python/base.cpp
void pybind_logging(py::module_ &m); // python/logging.cpp
void pybind_initializers(py::module_ &m); // python/initializers.cpp
void pybind_problems(py::module_ &m); // python/problems.cpp
void pybind_archive(py::module_ &m); // python/archive.cpp
void pybind_acceptation_criteria(py::module_ &m); // python/acceptation_criteria.cpp
void pybind_ga(py::module_ &m); // python/ga.cpp
void pybind_gomea(py::module_ &m); // python/gomea.cpp
void pybind_running(py::module_ &m); // python/running.cpp
void pybind_sim(py::module_ &m); // python/sim.cpp
void pybind_sim_gomea(py::module_ &m); // python/sim-gomea.cpp
void pybind_ecga(py::module_ &m); // python/ecga.cpp
void pybind_sim_ga(py::module_ &m); // python/sim-ga.cpp

PYBIND11_MODULE(ealib, m)
{
    m.doc() = "A library for evolutionary algorithms";

    pybind_utilities(m);
    pybind_base(m);
    pybind_logging(m);
    pybind_initializers(m);
    pybind_problems(m);
    pybind_archive(m);
    // Note: Weird bug: putting this in its own file breaks initialization...
    py::class_<BruteforceArchive, IArchive, std::shared_ptr<BruteforceArchive>>(m, "BruteforceArchive")
        .def(py::init<std::vector<size_t>>(), py::arg("objective_indices"))
        .def("try_add", &BruteforceArchive::try_add)
        .def("get_archived", &BruteforceArchive::get_archived);
    pybind_acceptation_criteria(m);
    pybind_ga(m);
    pybind_gomea(m);
    pybind_running(m);   
    pybind_sim(m);
    pybind_sim_gomea(m);
    pybind_ecga(m);
    pybind_sim_ga(m);
    
    py::register_exception<assertion_failure>(m, "AssertionFailure");
}