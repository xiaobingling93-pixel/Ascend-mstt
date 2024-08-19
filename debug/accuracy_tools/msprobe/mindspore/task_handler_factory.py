from msprobe.core.common.const import Const
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_tool_factory import DumpToolFactory
from msprobe.mindspore.overflow_check.overflow_check_tool_factory import OverflowCheckToolFactory
from msprobe.mindspore.free_benchmark.self_check_tool_factory import SelfCheckToolFactory


class TaskHandlerFactory:
    tasks = {
        Const.TENSOR: DumpToolFactory,
        Const.STATISTICS: DumpToolFactory,
        Const.OVERFLOW_CHECK: OverflowCheckToolFactory,
        Const.FREE_BENCHMARK: SelfCheckToolFactory
    }

    @staticmethod
    def create(config: DebuggerConfig):
        task = TaskHandlerFactory.tasks.get(config.task)
        if not task:
            raise Exception("Valid task is needed.")
        handler = task.create(config)
        if not handler:
            raise Exception("Can not find task handler")
        return handler
