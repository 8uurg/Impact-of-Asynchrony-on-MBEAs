#include "logging.hpp"
#include "pybindi.h"

#include "archive.hpp"
#include <pybind11/stl.h>

void pybind_archive(py::module_ &m)
{
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<IArchive, IDataUser, std::shared_ptr<IArchive>>(m, "IArchive");
    // Note: in python.cpp instead due to a weird bug.
    // py::class_<BruteforceArchive, IArchive, std::shared_ptr<BruteforceArchive>>(m, "BruteforceArchive")
    //     .def(py::init<std::vector<size_t>>(), py::arg("objective_indices"))
    //     .def("try_add", &BruteforceArchive::try_add)
    //     .def("get_archived", &BruteforceArchive::get_archived);
    py::class_<ArchivedLogger, ItemLogger, std::shared_ptr<ArchivedLogger>>(m, "ArchivedLogger").def(py::init());
    py::class_<LoggingArchive, IArchive, std::shared_ptr<LoggingArchive>>(m, "LoggingArchive")
        .def(py::init<std::shared_ptr<IArchive>,
                      std::shared_ptr<BaseLogger>,
                      std::optional<std::shared_ptr<ArchivedLogger>>>(),
             py::arg("archive"),
             py::arg("logger"),
             py::arg("archive_item_logger") = std::nullopt)
        .def("try_add", &LoggingArchive::try_add)
        .def("get_archived", &LoggingArchive::get_archived);

    py::class_<ImprovementTrackingArchive, IArchive, std::shared_ptr<ImprovementTrackingArchive>>(
        m, "ImprovementTrackingArchive")
        .def(py::init<std::shared_ptr<IArchive>, std::function<double()>, double, double>(),
             py::arg("archive"),
             py::arg("get_current_t"),
             py::arg("base") = 10,
             py::arg("factor") = 10);
}