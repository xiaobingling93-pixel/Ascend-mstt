import json
import os

from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.core.common.file_utils import FileOpen
from msprobe.core.common.const import Const
from msprobe.pytorch.hook_module.utils import get_ops
from msprobe.core.grad_probe.constant import level_adp
from msprobe.core.grad_probe.utils import check_numeral_list_ascend


class TensorConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.online_run_ut = json_config.get("online_run_ut", False)
        self.nfs_path = json_config.get("nfs_path", "")
        self.host = json_config.get("host", "")
        self.port = json_config.get("port", -1)
        self.tls_path = json_config.get("tls_path", "")
        self.check_config()
        self._check_file_format()
        self._check_tls_path_config()

    def _check_file_format(self):
        if self.file_format is not None and self.file_format not in ["npy", "bin"]:
            raise Exception("file_format is invalid")

    def _check_tls_path_config(self):
        if self.tls_path:
            if not os.path.exists(self.tls_path):
                raise Exception("tls_path: %s does not exist" % self.tls_path)
            if not os.path.exists(os.path.join(self.tls_path, "client.key")):
                raise Exception("tls_path does not contain client.key")
            if not os.path.exists(os.path.join(self.tls_path, "client.crt")):
                raise Exception("tls_path does not contain client.crt")


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
        self.overflow_nums = json_config.get("overflow_nums")
        self.check_mode = json_config.get("check_mode")
        self.check_overflow_config()

    def check_overflow_config(self):
        if self.overflow_nums is not None and not isinstance(self.overflow_nums, int):
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
    WrapApi = get_ops()

    def __init__(self, json_config):
        super().__init__(json_config)
        self.white_list = json_config.get("white_list", Const.DEFAULT_LIST)
        self.black_list = json_config.get("black_list", Const.DEFAULT_LIST)
        self.error_data_path = json_config.get("error_data_path", Const.DEFAULT_PATH)
        self.is_online = json_config.get("is_online", False)
        self.nfs_path = json_config.get("nfs_path", "")
        self.host = json_config.get("host", "")
        self.port = json_config.get("port", -1)
        self.rank_list = json_config.get("rank_list", Const.DEFAULT_LIST)
        self.tls_path = json_config.get("tls_path", "")
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

    @classmethod
    def check_nfs_path_config(cls, nfs_path):
        if nfs_path and not os.path.exists(nfs_path):
            raise Exception("nfs_path: %s does not exist" % nfs_path)

    @classmethod
    def check_tls_path_config(cls, tls_path):
        if tls_path:
            if not os.path.exists(tls_path):
                raise Exception("tls_path: %s does not exist" % tls_path)
            if not os.path.exists(os.path.join(tls_path, "server.key")):
                raise Exception("tls_path does not contain server.key")
            if not os.path.exists(os.path.join(tls_path, "server.crt")):
                raise Exception("tls_path does not contain server.crt")

    def check_run_ut_config(self):
        RunUTConfig.check_filter_list_config(Const.WHITE_LIST, self.white_list)
        RunUTConfig.check_filter_list_config(Const.BLACK_LIST, self.black_list)
        RunUTConfig.check_error_data_path_config(self.error_data_path)
        RunUTConfig.check_nfs_path_config(self.nfs_path)
        RunUTConfig.check_tls_path_config(self.tls_path)


class GradToolConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.grad_level = json_config.get("grad_level", "L1")
        self.param_list = json_config.get("param_list", [])
        self.bounds = json_config.get("bounds", [-1, 0, 1])
        self._check_config()
    
    def _check_config(self):
        if self.grad_level not in level_adp.keys():
            raise Exception(f"grad_level must be one of {level_adp.keys()}")
        if not isinstance(self.param_list, list):
            raise Exception(f"param_list must be a list")
        check_numeral_list_ascend(self.bounds)


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
    elif task == Const.GRAD_PROBE:
        config_dic = json_config.get(Const.GRAD_PROBE, default_dic)
        return GradToolConfig(config_dic)
    else:
        return StatisticsConfig(default_dic)


def parse_json_config(json_file_path, task):
    if not json_file_path:
        config_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        json_file_path = os.path.join(config_dir, "config.json")
    with FileOpen(json_file_path, 'r') as file:
        json_config = json.load(file)
    common_config = CommonConfig(json_config)
    if task and task in Const.TASK_LIST:
        task_config = parse_task_config(task, json_config)
    else:
        task_config = parse_task_config(common_config.task, json_config)
    return common_config, task_config
