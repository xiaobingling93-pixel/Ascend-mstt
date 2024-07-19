import json
from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.core.common.file_check import FileOpen


class TensorConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.check_mode = None
        self.file_format = json_config.get("file_format")
        self.check_config()
        self._check_config()

    def _check_config(self):
        if self.data_mode is not None and len(self.data_mode) > 0:
            if len(self.data_mode) > 1 or self.data_mode[0] not in ["all", "input", "output"]:
                raise Exception("data_mode must be all, input or output")
        if self.file_format and self.file_format not in ["npy", "bin"]:
            raise Exception("file_format is invalid")


class StatisticsConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.file_format = None
        self.check_mode = None
        self.check_config()
        self._check_config()

    def _check_config(self):
        if self.data_mode is not None and len(self.data_mode) > 0:
            if len(self.data_mode) > 1 or self.data_mode[0] not in ["all", "input", "output"]:
                raise Exception("data_mode must be all, input or output")


class OverflowCheck(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.file_format = None
        self.check_mode = json_config.get("check_mode")
        self._check_config()

    def _check_config(self):
        if self.data_mode is not None and len(self.data_mode) > 0:
            if len(self.data_mode) > 1 or self.data_mode[0] not in ["all", "input", "output"]:
                raise Exception("data_mode must be all, input or output")
        if self.check_mode and self.check_mode not in ["all", "aicore", "atomic"]:
            raise Exception("check_mode is invalid")


def parse_common_config(json_config):
    return CommonConfig(json_config)


def parse_task_config(task, json_config):
    task_map = json_config[task]
    if not task_map:
        task_map = dict()
    if task == "tensor":
        return TensorConfig(task_map)
    elif task == "statistics":
        return StatisticsConfig(task_map)
    elif task == "overflow_check":
        return OverflowCheck(task_map)
    else:
        raise Exception("task is invalid.")


def parse_json_config(json_file_path):
    if not json_file_path:
        raise Exception("json file path is None")
    with FileOpen(json_file_path, 'r') as file:
        json_config = json.load(file)
    common_config = parse_common_config(json_config)
    if not common_config.task:
        common_config.task = "statistics"
    task_config = parse_task_config(common_config.task, json_config)
    return common_config, task_config
