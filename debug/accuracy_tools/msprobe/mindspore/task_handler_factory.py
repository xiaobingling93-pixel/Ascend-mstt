from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_tool_factory import DumpToolFactory
from msprobe.mindspore.overflow_check.overflow_check_tool_factory import OverflowCheckToolFactory


class TaskHandlerFactory:
    tasks = {
        "tensor": DumpToolFactory,
        "statistics": DumpToolFactory,
        "overflow_check": OverflowCheckToolFactory
    }

    @staticmethod
    def create(config: DebuggerConfig):
        task = TaskHandlerFactory.tasks.get(config.task)
        if not task:
            raise Exception("valid task is needed.")
        handler = task.create(config)
        if not handler:
            raise Exception("Can not find task handler")
        return handler
