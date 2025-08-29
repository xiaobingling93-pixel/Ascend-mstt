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
#include "ipc_monitor/utils.h"

namespace py = pybind11;


PYBIND11_MODULE(IPCMonitor_C, m) {
    m.def("init_dyno", [](int npu_id) -> bool {
        return dynolog_npu::ipc_monitor::PyDynamicMonitorProxy::GetInstance()->InitDyno(npu_id);
    }, py::arg("npu_id"));
    m.def("poll_dyno", []() -> std::string {
        return dynolog_npu::ipc_monitor::PyDynamicMonitorProxy::GetInstance()->PollDyno();
    });
    m.def("enable_dyno_npu_monitor", [](std::unordered_map<std::string, std::string>& config_map) -> void {
        dynolog_npu::ipc_monitor::PyDynamicMonitorProxy::GetInstance()->EnableMsptiMonitor(config_map);
    }, py::arg("config_map"));
    m.def("finalize_dyno", []() -> void {
        dynolog_npu::ipc_monitor::PyDynamicMonitorProxy::GetInstance()->FinalizeDyno();
    });
    m.def("set_parallel_group_info", [](std::string parallel_group_info) -> void {
        dynolog_npu::ipc_monitor::SetParallelGroupInfo(parallel_group_info);
    }, py::arg("parallel_group_info"));
}
