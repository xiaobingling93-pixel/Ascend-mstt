import json
import os

from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.core.common.file_check import FileOpen
from msprobe.core.common.const import Const
from msprobe.pytorch.hook_module.utils import WrapFunctionalOps, WrapTensorOps, WrapTorchOps


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


class RunUTConfig(BaseConfig):
    WrapApi = set(WrapFunctionalOps) | set(WrapTensorOps) | set(WrapTorchOps)
    def __init__(self, json_config):
        super().__init__(json_config)
        self.white_list = json_config.get("white_list", Const.DEFAULT_LIST)
        self.black_list = json_config.get("black_list", Const.DEFAULT_LIST)
        self.error_data_path = json_config.get("error_data_path", Const.DEFAULT_PATH)
        self.check_run_ut_config()

    @classmethod
    def check_filter_list_config(cls, key, filter_list):
        if not isinstance(filter_list, list):
            raise Exception("%s must be a list type" % key)
        if not all(isinstance(item, str) for item in filter_list):
            raise Exception("All elements in %s must be string type" % key)
        invalid_api = [item for item in filter_list if item not in cls.WrapApi]
        if invalid_api:
            raise Exception("Invalid api in %s: %s" % (key, invalid_api))

    @classmethod
    def check_error_data_path_config(cls, error_data_path):
        if not os.path.exists(error_data_path):
            raise Exception("error_data_path: %s does not exist" % error_data_path)
        
    def check_run_ut_config(self):
        RunUTConfig.check_filter_list_config(Const.WHITE_LIST, self.white_list)
        RunUTConfig.check_filter_list_config(Const.BLACK_LIST, self.black_list)
        RunUTConfig.check_error_data_path_config(self.error_data_path)


def parse_task_config(task, json_config):
    default_dic = {}
    if task == Const.TENSOR:
        config_dic = json_config.get(Const.TENSOR, default_dic)
        return TensorConfig(config_dic)
    elif task == Const.STATISTICS:
        config_dic = json_config.get(Const.STATISTICS, default_dic)
        return StatisticsConfig(config_dic)
    elif task == Const.OVERFLOW_CHECK:
        config_dic = json_config.get(Const.OVERFLOW_CHECK, default_dic)
        return OverflowCheckConfig(config_dic)
    elif task == Const.FREE_BENCHMARK:
        config_dic = json_config.get(Const.FREE_BENCHMARK, default_dic)
        return FreeBenchmarkCheckConfig(config_dic)
    elif task == Const.RUN_UT:
        config_dic = json_config.get(Const.RUN_UT, default_dic)
        return RunUTConfig(config_dic)
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
