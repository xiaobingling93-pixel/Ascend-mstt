from msprobe.core.common.const import Const
from msprobe.mindspore.common.const import Const as MsConst
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
        if config.execution_mode == MsConst.PYNATIVE_MODE and config.task != Const.FREE_BENCHMARK:
            raise Exception("Current Task can't run in pynative mode.")
        task = TaskHandlerFactory.tasks.get(config.task)
        if not task:
            raise Exception("valid task is needed.")
        handler = task.create(config)
        if not handler:
            raise Exception("Can not find task handler")
        return handler
