#include "pybindi.h"

#include "logging.hpp"

void pybind_logging(py::module_ &m)
{
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<ItemLogger, IDataUser, std::shared_ptr<ItemLogger>>(m, "ItemLogger");
    py::class_<SequencedItemLogger, ItemLogger, std::shared_ptr<SequencedItemLogger>>(m, "SequencedItemLogger")
        .def(py::init<std::vector<std::shared_ptr<ItemLogger>>>(), py::arg("subloggers"));
    py::class_<GenotypeCategoricalLogger, ItemLogger, std::shared_ptr<GenotypeCategoricalLogger>>(m, "GenotypeCategoricalLogger")
        .def(py::init<>());
    py::class_<ObjectiveLogger, ItemLogger, std::shared_ptr<ObjectiveLogger>>(m, "ObjectiveLogger")
        .def(py::init<>());
    py::class_<NumEvaluationsLogger, ItemLogger, std::shared_ptr<NumEvaluationsLogger>>(m, "NumEvaluationsLogger")
        .def(py::init<std::shared_ptr<Limiter>>(), py::arg("limiter"));
    py::class_<WallTimeLogger, ItemLogger, std::shared_ptr<WallTimeLogger>>(m, "WallTimeLogger")
        .def(py::init<std::shared_ptr<Limiter>>(), py::arg("limiter"));
    py::class_<SolutionIndexLogger, ItemLogger, std::shared_ptr<SolutionIndexLogger>>(m, "SolutionIndexLogger")
        .def(py::init<>());

    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<BaseLogger, IDataUser, std::shared_ptr<BaseLogger>>(m, "BaseLogger");
    py::class_<CSVLogger, BaseLogger, std::shared_ptr<CSVLogger>>(m, "CSVLogger")
        .def(py::init<std::filesystem::path, std::shared_ptr<ItemLogger>>(), py::arg("out_path"), py::arg("item_logger"));

    
}