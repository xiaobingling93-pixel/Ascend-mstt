/*
 * Copyright (C) 2025-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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