import json
import os

from ..core.common_config import CommonConfig, BaseConfig
from ..core.file_check_util import FileOpen
from ..core.utils import Const


# 特定任务配置类
class TensorConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.check_config()
        self._check_file_format()

    def _check_file_format(self):
        if self.file_format is not None and self.file_format not in ["npy", "bin"]:
            raise Exception("file_format is invalid")


class StatisticsConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.check_config()
        self._check_summary_mode()

    def _check_summary_mode(self):
        if self.summary_mode and self.summary_mode not in ["statistics", "md5"]:
            raise Exception("summary_mode is invalid")


class OverflowCheckConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.overflow_num = json_config.get("overflow_nums")
        self.check_mode = json_config.get("check_mode")
        self.check_overflow_config()

    def check_overflow_config(self):
        if self.overflow_num is not None and not isinstance(self.overflow_num, int):
            raise Exception("overflow_num is invalid")
        if self.check_mode is not None and self.check_mode not in ["all", "aicore", "atomic"]:
            raise Exception("check_mode is invalid")


class FreeBenchmarkCheckConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.fuzz_device = json_config.get("fuzz_device")
        self.pert_mode = json_config.get("pert_mode")
        self.handler_type = json_config.get("handler_type")
        self.fuzz_level = json_config.get("fuzz_level")
        self.fuzz_stage = json_config.get("fuzz_stage")
        self.if_preheat = json_config.get("if_preheat")
        self.preheat_step = json_config.get("preheat_step")
        self.max_sample = json_config.get("max_sample")
        self.check_freebenchmark_config()

    def check_freebenchmark_config(self):
        if self.if_preheat and self.handler_type == "fix":
            raise Exception("Preheating is not supported in fix handler type")
        if self.preheat_step and self.preheat_step == 0:
            raise Exception("preheat_step cannot be 0")

def parse_task_config(task, json_config):
    default_dic = {}
    if task == Const.TENSOR:
        config_dic = json_config.get(Const.TENSOR) if json_config.get(Const.TENSOR) else default_dic
        return TensorConfig(config_dic)
    elif task == Const.STATISTICS:
        config_dic = json_config.get(Const.STATISTICS) if json_config.get(Const.STATISTICS) else default_dic
        return StatisticsConfig(config_dic)
    elif task == Const.OVERFLOW_CHECK:
        config_dic = json_config.get(Const.OVERFLOW_CHECK) if json_config.get(Const.OVERFLOW_CHECK) else default_dic
        return OverflowCheckConfig(config_dic)
    elif task == Const.FREE_BENCHMARK:
        config_dic = json_config.get(Const.FREE_BENCHMARK) if json_config.get(Const.FREE_BENCHMARK) else default_dic
        return FreeBenchmarkCheckConfig(config_dic)
    else:
        return StatisticsConfig(default_dic)


def parse_json_config(json_file_path, task):
    if not json_file_path:
        config_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        json_file_path = os.path.join(os.path.join(config_dir, "config"), "config.json")
    with FileOpen(json_file_path, 'r') as file:
        json_config = json.load(file)
    common_config = CommonConfig(json_config)
    if task and task in Const.TASK_LIST:
        task_config = parse_task_config(task, json_config)
    else:
        task_config = parse_task_config(common_config.task, json_config)
    return common_config, task_config
