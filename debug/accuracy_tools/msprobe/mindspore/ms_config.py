import json

from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.core.common.file_check import FileOpen
from msprobe.core.common.const import Const, MsFreeBenchmarkConst
from msprobe.mindspore.common.log import logger


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
        if self.summary_mode and self.summary_mode not in ["statistics", "md5"]:
            raise Exception("summary_mode is invalid")


class OverflowCheckConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.data_mode = ["all"]
        self._check_config()

    def _check_config(self):
        if self.overflow_nums is not None and not isinstance(self.overflow_nums, int):
            raise Exception("overflow_nums is invalid, it should be an integer")
        if self.overflow_nums is not None and self.overflow_nums != -1 and self.overflow_nums <= 0:
            raise Exception("overflow_nums should be -1 or positive integer")
        if self.check_mode and self.check_mode not in ["all", "aicore", "atomic"]:
            raise Exception("check_mode is invalid")


class FreeBenchmarkConfig(BaseConfig):
    def __init__(self, task_config):
        super().__init__(task_config)
        self._check_config()

    def _check_config(self):
        if self.fuzz_device and self.fuzz_device not in MsFreeBenchmarkConst.DEVICE_LIST:
            raise Exception("fuzz_device must be npu or empty")
        if self.pert_mode and self.pert_mode not in MsFreeBenchmarkConst.PERT_TYPE_LIST:
            raise Exception("pert_mode must be improve_precision, add_noise, bit_noise , no_change or empty")
        if self.handler_type and self.handler_type not in MsFreeBenchmarkConst.HANDLER_TYPE_LIST:
            raise Exception("handler_type must be check, fix or empty")
        if self.fuzz_level and self.fuzz_level not in MsFreeBenchmarkConst.DUMP_LEVEL_LIST:
            raise Exception("fuzz_level must be L1 or empty")
        if self.fuzz_stage and self.fuzz_stage not in MsFreeBenchmarkConst.STAGE_LIST:
            raise Exception("fuzz_stage must be forward or empty")
        if self.if_preheat or self.preheat_step or self.max_sample:
            logger.warning("'if_preheat', 'preheat_step' and 'max_sample' settings "
                           "are not supported for mindspore free benchmark task.")


TaskDict = {
    Const.TENSOR: TensorConfig,
    Const.STATISTICS: StatisticsConfig,
    Const.OVERFLOW_CHECK: OverflowCheckConfig,
    Const.FREE_BENCHMARK: FreeBenchmarkConfig
}


def parse_common_config(json_config):
    return CommonConfig(json_config)


def parse_task_config(task, json_config):
    task_map = json_config.get(task)
    if not task_map:
        task_map = dict()
    if task not in TaskDict:
        raise Exception("task is invalid.")
    return TaskDict.get(task)(task_map)


def parse_json_config(json_file_path):
    if not json_file_path:
        raise Exception("json file path is None")
    with FileOpen(json_file_path, 'r') as file:
        json_config = json.load(file)
    common_config = parse_common_config(json_config)
    if not common_config.task:
        common_config.task = Const.STATISTICS
    task_config = parse_task_config(common_config.task, json_config)
    return common_config, task_config
