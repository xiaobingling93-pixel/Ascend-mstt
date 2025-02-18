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

import torch

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.pytorch.common.log import logger


class DebuggerConfig:
    def __init__(self, common_config, task_config, task, dump_path, level):
        self.dump_path = dump_path if dump_path else common_config.dump_path
        self.task = task or common_config.task or Const.STATISTICS
        self.rank = common_config.rank if common_config.rank else []
        self.step = common_config.step if common_config.step else []
        self.level = level or common_config.level or Const.LEVEL_L1
        self.enable_dataloader = common_config.enable_dataloader
        self.scope = task_config.scope if task_config.scope else []
        self.list = task_config.list if task_config.list else []
        self.data_mode = task_config.data_mode if task_config.data_mode else ["all"]
        self.summary_mode = task_config.summary_mode if task_config.summary_mode else Const.STATISTICS
        self.overflow_nums = task_config.overflow_nums if task_config.overflow_nums else 1
        self.framework = Const.PT_FRAMEWORK
        self.async_dump = common_config.async_dump if common_config.async_dump else False

        if self.task == Const.FREE_BENCHMARK:
            self.fuzz_device = task_config.fuzz_device
            self.handler_type = task_config.handler_type
            self.pert_mode = task_config.pert_mode
            self.fuzz_level = task_config.fuzz_level
            self.fuzz_stage = task_config.fuzz_stage
            self.preheat_config = {
                "if_preheat": task_config.if_preheat,
                "preheat_step": task_config.preheat_step,
                "max_sample": task_config.max_sample
            }

        self.online_run_ut = False
        if self.task == Const.TENSOR:
            # dump api tensor and collaborate with online run_ut
            self.online_run_ut = task_config.online_run_ut if task_config.online_run_ut else False
            self.nfs_path = task_config.nfs_path if task_config.nfs_path else ""
            self.tls_path = task_config.tls_path if task_config.tls_path else ""
            self.host = task_config.host if task_config.host else ""
            self.port = task_config.port if task_config.port else -1
            self.online_run_ut_recompute = task_config.online_run_ut_recompute \
                if isinstance(task_config.online_run_ut_recompute, bool) else False

        self.check()

        if self.level == Const.LEVEL_L2:
            self.is_backward_kernel_dump = False
            self._check_and_adjust_config_with_l2()

    def check_kwargs(self):
        if self.task and self.task not in Const.TASK_LIST:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"The task <{self.task}> is not in the {Const.TASK_LIST}.")
        if self.level and self.level not in Const.LEVEL_LIST:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"The level <{self.level}> is not in the {Const.LEVEL_LIST}.")
        if not self.dump_path:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"The dump_path not found.")
        if not isinstance(self.async_dump, bool):
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"The parameters async_dump should be bool.")
        if self.async_dump and self.task == Const.TENSOR and not self.list:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"The parameters async_dump is true in tensor task, the parameters list cannot be "
                                   f"empty.")
        if self.task == Const.STRUCTURE and self.level not in [Const.LEVEL_L0, Const.LEVEL_MIX]:
            logger.warning_on_rank_0(
                f"When the task is set to structure, the level should be one of {[Const.LEVEL_L0, Const.LEVEL_MIX]}. "
                f"If not, the default level is {Const.LEVEL_MIX}."
            )
            self.level = Const.LEVEL_MIX

    def check(self):
        self.check_kwargs()
        return True

    def check_model(self, instance, start_model):
        if self.level not in [Const.LEVEL_L0, Const.LEVEL_MIX]:
            if instance.model is not None or start_model is not None:
                logger.info_on_rank_0(
                    f"The current level is not L0 or mix level, so the model parameters will not be used.")
            return
        if start_model is None and instance.model is None:
            logger.error_on_rank_0(
                f"For level {self.level}, PrecisionDebugger or start interface must receive a 'model' parameter.")
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, f"missing the parameter 'model'")

        instance.model = start_model if start_model is not None else instance.model
        if isinstance(instance.model, torch.nn.Module):
            return

        error_model = None
        if isinstance(instance.model, (list, tuple)):
            for model in instance.model:
                if not isinstance(model, torch.nn.Module):
                    error_model = model
                    break
        else:
            error_model = instance.model

        if error_model is not None:
            error_info = (f"The 'model' parameter must be a torch.nn.Module or list[torch.nn.Module] "
                          f"type, currently there is a {type(error_model)} type.")
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, error_info)

    def _check_and_adjust_config_with_l2(self):
        if self.scope:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"When level is set to L2, the scope cannot be configured.")
        if not self.list or len(self.list) != 1:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"When level is set to L2, the list must be configured as a list with one api name.")
        api_name = self.list[0]
        if api_name.endswith(Const.BACKWARD):
            self.is_backward_kernel_dump = True
            api_forward_name = api_name[:-len(Const.BACKWARD)] + Const.FORWARD
            self.list.append(api_forward_name)
