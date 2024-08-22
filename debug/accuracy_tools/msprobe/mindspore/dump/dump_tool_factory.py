from msprobe.mindspore.common.const import Const
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.kernel_kbyk_dump import KernelKbykDump
from msprobe.mindspore.dump.kernel_graph_dump import KernelGraphDump


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
