#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ipc_monitor/PyDynamicMonitorProxy.h"

namespace py = pybind11;

PYBIND11_MODULE(IPCMonitor, m) {
   py::class_<dynolog_npu::ipc_monitor::PyDynamicMonitorProxy>(m, "PyDynamicMonitorProxy")
        .def(py::init<>())
        .def("init_dyno", &dynolog_npu::ipc_monitor::PyDynamicMonitorProxy::InitDyno, py::arg("npuId"))
        .def("poll_dyno", &dynolog_npu::ipc_monitor::PyDynamicMonitorProxy::PollDyno)
        .def("enable_dyno_npu_monitor", &dynolog_npu::ipc_monitor::PyDynamicMonitorProxy::EnableMsptiMonitor, py::arg("cfg_map"))
        .def("finalize_dyno", &dynolog_npu::ipc_monitor::PyDynamicMonitorProxy::FinalizeDyno);
}