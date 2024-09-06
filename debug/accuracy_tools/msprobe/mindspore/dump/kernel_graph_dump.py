import os
import json
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.common.log import logger
from msprobe.core.common.file_utils import FileOpen, create_directory


class KernelGraphDump:
    def __init__(self, config: DebuggerConfig):
        self.dump_json = dict()
        self.dump_json["common_dump_settings"] = dict()
        self.dump_json["common_dump_settings"]["dump_mode"] = 0
        self.dump_json["common_dump_settings"]["path"] = ""
        self.dump_json["common_dump_settings"]["net_name"] = "Net"
        self.dump_json["common_dump_settings"]["iteration"] = "all"
        self.dump_json["common_dump_settings"]["saved_data"] = "statistic"
        self.dump_json["common_dump_settings"]["input_output"] = 0
        self.dump_json["common_dump_settings"]["kernels"] = []
        self.dump_json["common_dump_settings"]["support_device"] = [0, 1, 2, 3, 4, 5, 6, 7]
        self.dump_json["common_dump_settings"]["op_debug_mode"] = 0
        self.dump_json["common_dump_settings"]["file_format"] = "npy"

        if len(config.list) > 0:
            self.dump_json["common_dump_settings"]["dump_mode"] = 1
            self.dump_json["common_dump_settings"]["kernels"] = config.list
        self.dump_json["common_dump_settings"]["path"] = config.dump_path
        if len(config.step) > 0:
            step_str = ""
            for s in config.step:
                step_str += (str(s) + '|')
            self.dump_json["common_dump_settings"]["iteration"] = step_str[:-1]
        if len(config.rank) > 0:
            self.dump_json["common_dump_settings"]["support_device"] = config.rank
        if config.task == "tensor":
            self.dump_json["common_dump_settings"]["saved_data"] = "tensor"
            self.dump_json["common_dump_settings"]["file_format"] = config.file_format
        if len(config.data_mode) == 1:
            if config.data_mode[0] == "input":
                self.dump_json["common_dump_settings"]["input_output"] = 1
            if config.data_mode[0] == "output":
                self.dump_json["common_dump_settings"]["input_output"] = 2

    def handle(self):
        if os.getenv("GRAPH_OP_RUN") == "1":
            raise Exception("Must run in graph mode, not kbk mode")
        json_path = self.dump_json["common_dump_settings"]["path"]
        create_directory(json_path)
        json_path = os.path.join(json_path, "kernel_graph_dump.json")
        with FileOpen(json_path, 'w') as f:
            json.dump(self.dump_json, f)
        logger.info(json_path + " has been created.")
        os.environ["MINDSPORE_DUMP_CONFIG"] = json_path
        if self.dump_json["common_dump_settings"]["dump_mode"] == 0:
            if self.dump_json["common_dump_settings"]["iteration"] != "all" or \
               len(self.dump_json["common_dump_settings"]["kernels"]) == 0:
                os.environ["MS_ACL_DUMP_CFG_PATH"] = json_path
        else:
            if "MS_ACL_DUMP_CFG_PATH" in os.environ:
                del os.environ["MS_ACL_DUMP_CFG_PATH"]
