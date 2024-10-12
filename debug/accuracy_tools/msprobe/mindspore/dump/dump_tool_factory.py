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
from msprobe.mindspore.dump.kernel_graph_dump import KernelGraphDump
from msprobe.mindspore.dump.kernel_kbyk_dump import KernelKbykDump


class DumpToolFactory:
    tools = {
        Const.CELL: {
            Const.GRAPH_KBYK_MODE: None,
            Const.GRAPH_GE_MODE: None,
            Const.PYNATIVE_MODE: None
        },
        Const.API: {
            Const.GRAPH_KBYK_MODE: None,
            Const.GRAPH_GE_MODE: None,
            Const.PYNATIVE_MODE: None
        },
        Const.KERNEL: {
            Const.GRAPH_KBYK_MODE: KernelKbykDump,
            Const.GRAPH_GE_MODE: KernelGraphDump,
            Const.PYNATIVE_MODE: KernelKbykDump
        }
    }

    @staticmethod
    def create(config: DebuggerConfig):
        tool = DumpToolFactory.tools.get(config.level)
        if not tool:
            raise Exception("Valid level is needed.")
        tool = tool.get(config.execution_mode)
        if not tool:
            raise Exception(f"Data dump is not supported in {config.execution_mode} mode "
                            f"when dump level is {config.level}.")
        return tool(config)
