from msprobe.core.common.const import MsConst
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.free_benchmark.api_pynative_self_check import ApiPyNativeSelFCheck


class SelfCheckToolFactory:
    tools = {
        MsConst.CELL: {
            MsConst.GRAPH_KBYK_MODE: None,
            MsConst.GRAPH_GE_MODE: None,
            MsConst.PYNATIVE_MODE: None
        },
        MsConst.API: {
            MsConst.GRAPH_KBYK_MODE: None,
            MsConst.GRAPH_GE_MODE: None,
            MsConst.PYNATIVE_MODE: ApiPyNativeSelFCheck
        },
        MsConst.KERNEL: {
            MsConst.GRAPH_KBYK_MODE: None,
            MsConst.GRAPH_GE_MODE: None,
            MsConst.PYNATIVE_MODE: None
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
