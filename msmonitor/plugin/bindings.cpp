#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ipc_monitor/PyDynamicMonitorProxy.h"

namespace py = pybind11;

void init_IPCMonitor(PyObject *module) {
    py::class_<dynolog_npu::ipc_monitor::PyDynamicMonitorProxy>(module, "PyDynamicMonitorProxy")
        .def(py::init<>())
        .def("init_dyno", &dynolog_npu::ipc_monitor::PyDynamicMonitorProxy::InitDyno, py::arg("npuId"))
        .def("poll_dyno", &dynolog_npu::ipc_monitor::PyDynamicMonitorProxy::PollDyno)
        .def("enable_dyno_npu_monitor", &dynolog_npu::ipc_monitor::PyDynamicMonitorProxy::EnableMsptiMonitor, py::arg("cfg_map"))
        .def("finalize_dyno", &dynolog_npu::ipc_monitor::PyDynamicMonitorProxy::FinalizeDyno);
}

static PyMethodDef g_moduleMethods[] = {};

static struct PyModuleDef ipcMonitor_module = {
    PyModuleDef_HEAD_INIT,
    "IPCMonitor",
    nullptr,
    -1,
    g_moduleMethods
};

PyMODINIT_FUNC PyInit_IPCMonitor(void) {
    PyObject* m = PyModule_Create(&ipcMonitor_module);
    init_IPCMonitor(m);
    return m;
}