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

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.file_utils import create_directory
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.core.common.log import logger


class DebuggerConfig:
    def __init__(self, common_config, task_config):
        self.execution_mode = None
        self.dump_path = common_config.dump_path
        self.task = common_config.task
        self.rank = [] if not common_config.rank else common_config.rank
        self.step = [] if not common_config.step else common_config.step
        common_config.level = Const.LEVEL_L1 if not common_config.level else common_config.level
        self.level = MsConst.TOOL_LEVEL_DICT.get(common_config.level, MsConst.API)
        self.level_ori = common_config.level
        self.list = [] if not task_config.list else task_config.list
        self.scope = [] if not task_config.scope else task_config.scope
        self.data_mode = [Const.ALL] if not task_config.data_mode else task_config.data_mode
        self.file_format = task_config.file_format
        self.overflow_nums = 1 if not task_config.overflow_nums else task_config.overflow_nums
        self.check_mode = task_config.check_mode
        self.framework = Const.MS_FRAMEWORK
        self.summary_mode = task_config.summary_mode
        self.async_dump = common_config.async_dump if common_config.async_dump else False
        self.check()
        create_directory(self.dump_path)

        if self.task == Const.FREE_BENCHMARK:
            self.pert_type = (FreeBenchmarkConst.DEFAULT_PERT_TYPE
                              if not task_config.pert_mode else task_config.pert_mode)
            self.handler_type = (FreeBenchmarkConst.DEFAULT_HANDLER_TYPE
                                 if not task_config.handler_type else task_config.handler_type)
            self.stage = FreeBenchmarkConst.DEFAULT_STAGE if not task_config.fuzz_stage else task_config.fuzz_stage
            if self.handler_type == FreeBenchmarkConst.FIX and \
                    self.pert_type != FreeBenchmarkConst.DEFAULT_PERT_TYPE:
                raise ValueError("pert_mode must be improve_precision or empty when handler_type is fix, "
                                 f"but got {self.pert_type}.")
            if self.stage == Const.BACKWARD and self.handler_type == FreeBenchmarkConst.FIX:
                raise ValueError("handler_type must be check or empty when fuzz_stage is backward, "
                                 f"but got {self.handler_type}.")
            self.dump_level = FreeBenchmarkConst.DEFAULT_DUMP_LEVEL

    def check(self):
        if not self.dump_path:
            raise Exception("Dump path is empty.")
        self.dump_path = os.path.abspath(self.dump_path)
        if not self.task:
            self.task = "statistics"
        if not self.level:
            raise Exception("level must be L0, L1 or L2")
        if not self.file_format:
            self.file_format = "npy"
        if not self.check_mode:
            self.check_mode = "all"
        if not isinstance(self.async_dump, bool):
            raise Exception("The parameters async_dump should be bool.")
        if self.async_dump and self.task == Const.TENSOR and not self.list:
            raise Exception("The parameters async_dump is true in tensor task, the parameters list cannot be empty.")
        if self.task == Const.STRUCTURE and self.level_ori not in [Const.LEVEL_L0, Const.LEVEL_MIX]:
            logger.warning_on_rank_0(
                f"When the task is set to structure, the level should be one of {[Const.LEVEL_L0, Const.LEVEL_MIX]}. "
                f"If not, the default level is {Const.LEVEL_MIX}."
            )
            self.level_ori = Const.LEVEL_MIX
        return True

    def check_config_with_l2(self):
        if self.level_ori != Const.LEVEL_L2:
            return
        if self.task != Const.TENSOR:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"When level is set to L2, the task must be set to tensor.")
        if self.scope:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"When level is set to L2, the scope cannot be configured.")
        if not self.list or len(self.list) != 1:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"When level is set to L2, the list must be configured as a list with one api name.")
