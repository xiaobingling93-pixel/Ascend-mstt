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
