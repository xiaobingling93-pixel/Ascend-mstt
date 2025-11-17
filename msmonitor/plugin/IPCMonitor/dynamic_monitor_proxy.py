# Copyright (c) 2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import importlib
from .singleton import Singleton
from .utils import get_parallel_group_info

so_path = os.path.join(os.path.dirname(__file__), "lib64")
sys.path.append(os.path.realpath(so_path))
ipcMonitor_C_module = importlib.import_module("IPCMonitor_C")


@Singleton
class PyDynamicMonitorProxy:

    @classmethod
    def init_dyno(cls, npu_id: int):
        return ipcMonitor_C_module.init_dyno(npu_id)

    @classmethod
    def poll_dyno(cls):
        return ipcMonitor_C_module.poll_dyno()

    @classmethod
    def enable_dyno_npu_monitor(cls, config_map: dict):
        if str(config_map.get("NPU_MONITOR_STOP")).lower() in ("true", "1"):
            ipcMonitor_C_module.set_cluster_config_data({"parallel_group_info": get_parallel_group_info()})
        ipcMonitor_C_module.enable_dyno_npu_monitor(config_map)

    @classmethod
    def finalize_dyno(cls):
        ipcMonitor_C_module.finalize_dyno()

    @classmethod
    def update_profiler_status(cls, status: dict):
        ipcMonitor_C_module.update_profiler_status(status)
