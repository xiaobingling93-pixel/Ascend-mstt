from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.overflow_check.kernel_graph_overflow_check import KernelGraphOverflowCheck


class OverflowCheckToolFactory:
    tools = {
        "cell": {
            "kbk": None,
            "graph": None,
            "pynative": None
        },
        "api": {
            "kbk": None,
            "graph": None,
            "pynative": None
        },
        "kernel": {
            "kbk": None,
            "graph": KernelGraphOverflowCheck,
            "pynative": None
        }
    }

    @staticmethod
    def create(config: DebuggerConfig):
        tool = OverflowCheckToolFactory.tools.get(config.level)
        if not tool:
            raise Exception("valid level is needed.")
        tool = tool.get("graph")
        if not tool:
            raise Exception("Overflow check in not supported in this mode.")
        return tool(config)
