# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

from msprobe.mindspore.common.const import Const
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.free_benchmark.api_pynative_self_check import ApiPyNativeSelFCheck


class SelfCheckToolFactory:
    tools = {
        Const.CELL: {
            Const.GRAPH_KBYK_MODE: None,
            Const.GRAPH_GE_MODE: None,
            Const.PYNATIVE_MODE: None
        },
        Const.API: {
            Const.GRAPH_KBYK_MODE: None,
            Const.GRAPH_GE_MODE: None,
            Const.PYNATIVE_MODE: ApiPyNativeSelFCheck
        },
        Const.KERNEL: {
            Const.GRAPH_KBYK_MODE: None,
            Const.GRAPH_GE_MODE: None,
            Const.PYNATIVE_MODE: None
        }
    }

    @staticmethod
    def create(config: DebuggerConfig):
        tool = SelfCheckToolFactory.tools.get(config.level)
        if not tool:
            raise Exception(f"{config.level} is not supported.")
        tool = tool.get(config.execution_mode)
        if not tool:
            raise Exception(f"Task free_benchmark is not supported in this mode: {config.execution_mode}.")
        return tool(config)
