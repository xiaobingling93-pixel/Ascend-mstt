# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


from msprobe.mindspore.common.const import Const
from msprobe.core.common.log import logger
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.free_benchmark.api_pynative_self_check import ApiPyNativeSelfCheck


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
            Const.PYNATIVE_MODE: ApiPyNativeSelfCheck
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
            logger.error(f"{config.level} is not supported.")
            raise ValueError
        tool = tool.get(config.execution_mode)
        if not tool:
            logger.error(f"Task free_benchmark is not supported in this mode: {config.execution_mode}.")
            raise ValueError
        return tool(config)
