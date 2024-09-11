import os
import json

from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.common.log import logger
from msprobe.core.common.file_utils import FileOpen, create_directory
from msprobe.core.common.const import Const


class KernelKbykDump:
    COMMON_SETTINGS = "common_dump_settings"
    E2E_SETTINGS = "e2e_dump_settings"

    def __init__(self, config: DebuggerConfig):
        self.dump_json = dict()
        common_set = dict()
        e2e_set = dict()

        common_set = dict()
        common_set["dump_mode"] = 0
        common_set["path"] = ""
        common_set["net_name"] = "Net"
        common_set["iteration"] = "all"
        common_set["saved_data"] = "statistic"
        common_set["input_output"] = 0
        common_set["kernels"] = []
        common_set["support_device"] = [0, 1, 2, 3, 4, 5, 6, 7]
        e2e_set = dict()
        e2e_set["enable"] = True
        e2e_set["trans_flag"] = True

        if config.list:
            common_set["dump_mode"] = 1
            common_set["kernels"] = config.list
        common_set["path"] = config.dump_path
        if config.step:
            step_str = ""
            for s in config.step:
                step_str += (str(s) + '|')
            common_set["iteration"] = step_str[:-1]
        if config.rank:
            common_set["support_device"] = config.rank
        if config.task == Const.TENSOR:
            common_set["saved_data"] = Const.TENSOR
        if len(config.data_mode) == 1:
            if config.data_mode[0] == Const.INPUT:
                common_set["input_output"] = 1
            if config.data_mode[0] == Const.OUTPUT:
                common_set["input_output"] = 2

        self.dump_json[KernelKbykDump.COMMON_SETTINGS] = common_set
        self.dump_json[KernelKbykDump.E2E_SETTINGS] = e2e_set

    def handle(self):
        json_path = self.dump_json[KernelKbykDump.COMMON_SETTINGS]["path"]
        create_directory(json_path)
        json_path = os.path.join(json_path, "kernel_kbyk_dump.json")
        with FileOpen(json_path, 'w') as f:
            json.dump(self.dump_json, f)
        logger.info(json_path + " has been created.")

        os.environ["MINDSPORE_DUMP_CONFIG"] = json_path
        if "MS_ACL_DUMP_CFG_PATH" in os.environ:
            del os.environ["MS_ACL_DUMP_CFG_PATH"]
