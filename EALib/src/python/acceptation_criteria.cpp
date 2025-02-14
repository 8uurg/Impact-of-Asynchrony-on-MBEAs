//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "base.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "acceptation_criteria.hpp"

void pybind_acceptation_criteria(py::module_ &m)
{
    py::class_<DominationObjectiveAcceptanceCriterion,
               IPerformanceCriterion,
               std::shared_ptr<DominationObjectiveAcceptanceCriterion>>(m, "DominationObjectiveAcceptanceCriterion")
        .def(py::init<std::vector<size_t>>(), py::arg("objective_indices"));

    py::class_<SingleObjectiveAcceptanceCriterion,
               IPerformanceCriterion,
               std::shared_ptr<SingleObjectiveAcceptanceCriterion>>(m, "SingleObjectiveAcceptanceCriterion")
        .def(py::init<size_t>(), py::arg("objective") = 0);

    py::class_<SequentialCombineAcceptanceCriterion,
               IPerformanceCriterion,
               std::shared_ptr<SequentialCombineAcceptanceCriterion>>(m, "SequentialCombineAcceptanceCriterion")
        .def(py::init<std::vector<std::shared_ptr<IPerformanceCriterion>>, bool>(),
             py::arg("criteria"),
             py::arg("nondeterminable_is_equal") = false);

    py::class_<ThresholdAcceptanceCriterion, IPerformanceCriterion, std::shared_ptr<ThresholdAcceptanceCriterion>>(
        m, "ThresholdAcceptanceCriterion")
        .def(py::init<size_t, double>(), py::arg("objective"), py::arg("threshold"));

    py::class_<Scalarizer, IDataUser, std::shared_ptr<Scalarizer>>(m, "Scalarizer");

    py::class_<TschebysheffObjectiveScalarizer, Scalarizer, std::shared_ptr<TschebysheffObjectiveScalarizer>>(
        m, "TschebysheffObjectiveScalarizer")
        .def(py::init<std::vector<size_t>>(), py::arg("objective_indices"));

    py::class_<ScalarizationAcceptanceCriterion,
               IPerformanceCriterion,
               std::shared_ptr<ScalarizationAcceptanceCriterion>>(m, "ScalarizationAcceptanceCriterion")
        .def(py::init<std::shared_ptr<Scalarizer>>(), py::arg("scalarizer"));

    py::class_<FunctionAcceptanceCriterion, IPerformanceCriterion, std::shared_ptr<FunctionAcceptanceCriterion>>(
        m, "FunctionAcceptanceCriterion")
        .def(py::init<std::function<short(Population & pop, Individual & a, Individual & b)>>(), py::arg("criterion"));
}