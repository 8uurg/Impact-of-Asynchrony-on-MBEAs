#include "pybindi.h"

#include "initializers.hpp"

void pybind_initializers(py::module_ &m)
{
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<ISolutionInitializer, IDataUser, std::shared_ptr<ISolutionInitializer>>(m, "ISolutionInitializer");
    py::class_<CategoricalUniformInitializer, ISolutionInitializer, std::shared_ptr<CategoricalUniformInitializer>>(
        m, "CategoricalUniformInitializer")
        .def(py::init<>());
    py::class_<CategoricalProbabilisticallyCompleteInitializer,
               ISolutionInitializer,
               std::shared_ptr<CategoricalProbabilisticallyCompleteInitializer>>(
        m, "CategoricalProbabilisticallyCompleteInitializer")
        .def(py::init<>());
    // - performance criteria
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<IPerformanceCriterion, IDataUser, std::shared_ptr<IPerformanceCriterion>>(m, "IPerformanceCriterion");
    // - generational approaches interface
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<GenerationalApproach, IDataUser, std::shared_ptr<GenerationalApproach>>(m, "GenerationalApproach")
        .def("step", &GenerationalApproach::step)
        .def("getSolutionPopulation", &GenerationalApproach::getSolutionPopulation);

    py::class_<GenerationalApproachComparator, IDataUser, std::shared_ptr<GenerationalApproachComparator>>(
        m, "GenerationalApproachComparator")
        .def(py::init<>());
    py::class_<AverageFitnessComparator, GenerationalApproachComparator, std::shared_ptr<AverageFitnessComparator>>(
        m, "AverageFitnessComparator")
        .def(py::init<>());
}