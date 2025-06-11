# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

from msprobe.core.common.const import Const, FileCheckConst
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.file_utils import FileOpen, load_json, check_file_or_directory_path, FileChecker
from msprobe.core.common.log import logger
from msprobe.core.common.utils import is_int
from msprobe.core.common_config import BaseConfig, CommonConfig
from msprobe.core.grad_probe.constant import level_adp
from msprobe.core.grad_probe.utils import check_bounds
from msprobe.pytorch.free_benchmark.common.enums import (
    DeviceType,
    HandlerType,
    PytorchFreeBenchmarkConst,
)
from msprobe.pytorch.hook_module.utils import get_ops


class TensorConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.online_run_ut = json_config.get("online_run_ut", False)
        self.nfs_path = json_config.get("nfs_path", "")
        self.host = json_config.get("host", "")
        self.port = json_config.get("port", -1)
        self.tls_path = json_config.get("tls_path", "./")
        self.online_run_ut_recompute = json_config.get("online_run_ut_recompute", False)
        self.check_config()
        self._check_summary_mode()
        self._check_file_format()
        if self.online_run_ut:
            self._check_online_run_ut()

    def _check_file_format(self):
        if self.file_format is not None and self.file_format not in ["npy", "bin"]:
            raise Exception("file_format is invalid")

    def _check_online_run_ut(self):
        if not isinstance(self.online_run_ut, bool):
            raise Exception(f"online_run_ut: {self.online_run_ut} is invalid.")

        if not isinstance(self.online_run_ut_recompute, bool):
            raise Exception(f"online_run_ut_recompute: {self.online_run_ut_recompute} is invalid.")

        if self.nfs_path:
            check_file_or_directory_path(self.nfs_path, isdir=True)
            return

        if self.tls_path:
            check_file_or_directory_path(self.tls_path, isdir=True)
            check_file_or_directory_path(os.path.join(self.tls_path, "client.key"))
            check_file_or_directory_path(os.path.join(self.tls_path, "client.crt"))
            check_file_or_directory_path(os.path.join(self.tls_path, "ca.crt"))
            crl_path = os.path.join(self.tls_path, "crl.pem")
            if os.path.exists(crl_path):
                check_file_or_directory_path(crl_path)

        if not isinstance(self.host, str) or not re.match(Const.ipv4_pattern, self.host):
            raise Exception(f"host: {self.host} is invalid.")

        if not isinstance(self.port, int) or not (0 < self.port <= 65535):
            raise Exception(f"port: {self.port} is invalid, port range 0-65535.")


class StatisticsConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.check_config()
        self._check_summary_mode()

        self.tensor_list = json_config.get("tensor_list", [])
        self._check_str_list_config(self.tensor_list, "tensor_list")


class OverflowCheckConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)
        self.overflow_nums = json_config.get("overflow_nums")
        self.check_mode = json_config.get("check_mode")
        self.check_overflow_config()

    def check_overflow_config(self):
        if self.overflow_nums is not None and not is_int(self.overflow_nums):
            raise Exception("overflow_num is invalid")
        if self.overflow_nums is not None and self.overflow_nums != -1 and self.overflow_nums <= 0:
            raise Exception("overflow_nums should be -1 or positive integer")
        if self.check_mode is not None and self.check_mode not in ["all", "aicore", "atomic"]:
            raise Exception("check_mode is invalid")


class FreeBenchmarkCheckConfig(BaseConfig):

    def __init__(self, json_config):
        super().__init__(json_config)
        self.fuzz_device = json_config.get("fuzz_device", PytorchFreeBenchmarkConst.DEFAULT_DEVICE)
        self.pert_mode = json_config.get("pert_mode", PytorchFreeBenchmarkConst.DEFAULT_MODE)
        self.handler_type = json_config.get("handler_type", PytorchFreeBenchmarkConst.DEFAULT_HANDLER)
        self.fuzz_level = json_config.get("fuzz_level", PytorchFreeBenchmarkConst.DEFAULT_FUZZ_LEVEL)
        self.fuzz_stage = json_config.get("fuzz_stage", PytorchFreeBenchmarkConst.DEFAULT_FUZZ_STAGE)
        self.if_preheat = json_config.get("if_preheat", False)
        self.preheat_step = json_config.get("preheat_step", PytorchFreeBenchmarkConst.DEFAULT_PREHEAT_STEP)
        self.max_sample = json_config.get("max_sample", PytorchFreeBenchmarkConst.DEFAULT_PREHEAT_STEP)
        self.check_freebenchmark_config()

    def check_freebenchmark_config(self):
        self._check_pert_mode()
        self._check_fuzz_device()
        self._check_handler_type()
        self._check_fuzz_stage()
        self._check_fuzz_level()
        self._check_if_preheat()
        if self.handler_type == HandlerType.FIX:
            self._check_fix_config()
        if self.if_preheat:
            self._check_preheat_config()

    def _check_pert_mode(self):
        if self.pert_mode not in PytorchFreeBenchmarkConst.PERTURBATION_MODE_LIST:
            msg = (
                f"pert_mode is invalid, it should be one of"
                f" {PytorchFreeBenchmarkConst.PERTURBATION_MODE_LIST}"
            )
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )

    def _check_fuzz_device(self):
        if self.fuzz_device not in PytorchFreeBenchmarkConst.DEVICE_LIST:
            msg = (
                f"fuzz_device is invalid, it should be one of"
                f" {PytorchFreeBenchmarkConst.DEVICE_LIST}"
            )
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )
        if (self.fuzz_device == DeviceType.CPU) ^ (
                self.pert_mode in PytorchFreeBenchmarkConst.CPU_MODE_LIST
        ):
            msg = (
                f"You need to and can only set fuzz_device as {DeviceType.CPU} "
                f"when pert_mode in {PytorchFreeBenchmarkConst.CPU_MODE_LIST}"
            )
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )

    def _check_handler_type(self):
        if self.handler_type not in PytorchFreeBenchmarkConst.HANDLER_LIST:
            msg = (
                f"handler_type is invalid, it should be one of"
                f" {PytorchFreeBenchmarkConst.HANDLER_LIST}"
            )
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )

    def _check_fuzz_stage(self):
        if self.fuzz_stage not in PytorchFreeBenchmarkConst.FUZZ_STAGE_LIST:
            msg = (
                f"fuzz_stage is invalid, it should be one of"
                f" {PytorchFreeBenchmarkConst.FUZZ_STAGE_LIST}"
            )
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )

    def _check_fuzz_level(self):
        if self.fuzz_level not in PytorchFreeBenchmarkConst.FUZZ_LEVEL_LIST:
            msg = (
                f"fuzz_level is invalid, it should be one of"
                f" {PytorchFreeBenchmarkConst.FUZZ_LEVEL_LIST}"
            )
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )

    def _check_if_preheat(self):
        if not isinstance(self.if_preheat, bool):
            msg = "if_preheat is invalid, it should be a boolean"
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )

    def _check_preheat_config(self):
        if not is_int(self.preheat_step):
            msg = "preheat_step is invalid, it should be an integer"
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )
        if self.preheat_step <= 0:
            msg = "preheat_step must be greater than 0"
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )
        if not is_int(self.max_sample):
            msg = "max_sample is invalid, it should be an integer"
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )
        if self.max_sample <= 0:
            msg = "max_sample must be greater than 0"
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )

    def _check_fix_config(self):
        if self.if_preheat:
            msg = f"Preheating is not supported for {HandlerType.FIX} handler type"
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )
        if self.fuzz_stage not in PytorchFreeBenchmarkConst.FIX_STAGE_LIST:
            msg = (
                f"The fuzz_stage when opening {HandlerType.FIX} handler must be one of "
                f"{PytorchFreeBenchmarkConst.FIX_STAGE_LIST}"
            )
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )
        if self.pert_mode not in PytorchFreeBenchmarkConst.FIX_MODE_LIST:
            msg = (
                f"The pert_mode when opening {HandlerType.FIX} handler must be one of "
                f"{PytorchFreeBenchmarkConst.FIX_MODE_LIST}"
            )
            logger.error_log_with_exp(
                msg, MsprobeException(MsprobeException.INVALID_PARAM_ERROR, msg)
            )


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
        self.tls_path = json_config.get("tls_path", "./")
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
        if nfs_path:
            FileChecker(nfs_path, FileCheckConst.DIR, FileCheckConst.READ_ABLE).common_check()

    @classmethod
    def check_tls_path_config(cls, tls_path):
        if tls_path:
            FileChecker(tls_path, FileCheckConst.DIR, FileCheckConst.READ_ABLE).common_check()

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
        check_bounds(self.bounds)


class StructureConfig(BaseConfig):
    def __init__(self, json_config):
        super().__init__(json_config)


TaskDict = {
    Const.TENSOR: TensorConfig,
    Const.STATISTICS: StatisticsConfig,
    Const.OVERFLOW_CHECK: OverflowCheckConfig,
    Const.FREE_BENCHMARK: FreeBenchmarkCheckConfig,
    Const.RUN_UT: RunUTConfig,
    Const.GRAD_PROBE: GradToolConfig,
    Const.STRUCTURE: StructureConfig
}


def parse_task_config(task, json_config):
    task_map = json_config.get(task, dict())
    return TaskDict.get(task)(task_map)


def parse_json_config(json_file_path, task):
    if not json_file_path:
        config_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        json_file_path = os.path.join(config_dir, "config.json")
    json_config = load_json(json_file_path)
    common_config = CommonConfig(json_config)
    if task:
        task_config = parse_task_config(task, json_config)
    else:
        task_config = parse_task_config(common_config.task, json_config)
    return common_config, task_config
