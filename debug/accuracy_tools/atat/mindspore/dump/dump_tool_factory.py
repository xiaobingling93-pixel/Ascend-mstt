from atat.mindspore.debugger.debugger_config import DebuggerConfig
from atat.mindspore.dump.api_kbk_dump import ApiKbkDump
from atat.mindspore.dump.kernel_graph_dump import KernelGraphDump


class DumpToolFactory:
    tools = {
        "cell": {
            "kbk": None,
            "graph": None,
            "pynative": None
        },
        "api": {
            "kbk": ApiKbkDump,
            "graph": None,
            "pynative": None
        },
        "kernel": {
            "kbk": None,
            "graph": KernelGraphDump,
            "pynative": None
        }
    }

    @staticmethod
    def create(config: DebuggerConfig):
        tool = DumpToolFactory.tools.get(config.level)
        if not tool:
            raise Exception("valid level is needed.")
        if config.level == "api":
            tool = tool.get("kbk")
        elif config.level == "kernel":
            tool = tool.get("graph")
        elif config.level == "cell":
            raise Exception("Cell dump in not supported now.")
        if not tool:
            raise Exception("Data dump in not supported in this mode.")
        return tool(config)