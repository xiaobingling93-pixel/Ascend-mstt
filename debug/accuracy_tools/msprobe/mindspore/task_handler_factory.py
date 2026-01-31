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


from msprobe.core.common.const import Const
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_tool_factory import DumpToolFactory
from msprobe.mindspore.overflow_check.overflow_check_tool_factory import OverflowCheckToolFactory
from msprobe.mindspore.free_benchmark.self_check_tool_factory import SelfCheckToolFactory
from msprobe.mindspore.exception_dump.exception_dump_tool_factory import ExceptionDumpToolFactory


class TaskHandlerFactory:
    tasks = {
        Const.TENSOR: DumpToolFactory,
        Const.STATISTICS: DumpToolFactory,
        Const.OVERFLOW_CHECK: OverflowCheckToolFactory,
        Const.FREE_BENCHMARK: SelfCheckToolFactory,
        Const.EXCEPTION_DUMP: ExceptionDumpToolFactory
    }

    @staticmethod
    def create(config: DebuggerConfig, model=None):
        task = TaskHandlerFactory.tasks.get(config.task)
        if not task:
            raise Exception("Valid task is needed.")
        if task == DumpToolFactory:
            handler = task.create(config, model)
        else:
            handler = task.create(config)
        if not handler:
            raise Exception("Can not find task handler")
        return handler
