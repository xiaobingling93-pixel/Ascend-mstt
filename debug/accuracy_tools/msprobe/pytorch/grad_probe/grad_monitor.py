# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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
from collections import defaultdict

import torch
from msprobe.core.common.file_utils import remove_path, save_npy, write_csv, create_directory
from msprobe.core.grad_probe.constant import level_adp
from msprobe.core.grad_probe.utils import check_numeral_list_ascend, data_in_list_target
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import get_rank_id, print_rank_0
from msprobe.pytorch.grad_probe.grad_stat_csv import GradStatCsv

if int(torch.__version__.split('.')[0]) >= 2:
    from torch.optim.optimizer import register_optimizer_step_pre_hook


class GradientMonitor:

    def __init__(self, common_config, task_config):
        level = task_config.grad_level
        if level not in level_adp:
            raise Exception(f"level is valid, not in {level_adp.keys()}")
        self._level_adp = level_adp[level]
        self._param_list = task_config.param_list
        self._target_ranks = common_config.rank
        logger.info(f"target rank {self._target_ranks}")
        self._target_step = common_config.step
        logger.info(f"target step {self._target_step}")
        self._bounds = task_config.bounds
        check_numeral_list_ascend(self._bounds)
        self._output_path = common_config.dump_path
        if not os.path.exists(self._output_path):
            create_directory(self._output_path)
        else:
            logger.warning(f"the file in {self._output_path} will be deleted")
        self._step = -1
        self._param2name = defaultdict(str)

    @property
    def output_path(self):
        return self._output_path

    @staticmethod
    def save_grad_direction(param_name, grad, save_path):
        if not os.path.exists(save_path):
            create_directory(save_path)
        param_grad = grad.clone().detach()
        is_positive = param_grad > 0
        save_filepath = os.path.join(save_path, f"{param_name}.npy")
        save_npy(is_positive.cpu().numpy(), save_filepath)

    def monitor(self, model):
        print_rank_0("> parameter names:")
        for name, param in model.named_parameters():
            self._param2name[param] = name
            print_rank_0(f"\t{name}")
        setattr(self, "_rank", get_rank_id())
        if torch.distributed.is_initialized() and not data_in_list_target(getattr(self, "_rank"), self._target_ranks):
            return
        self._hook_optimizer()

    def _hook_optimizer(self):
        def optimizer_pre_step_hook(optimizer, args, kargs):
            self._step += 1
            logger.info(f"grad_probe: optimizer step {self._step}")
            if not data_in_list_target(self._step, self._target_step):
                return
            output_lines = []
            for param, param_name in self._param2name.items():
                if not data_in_list_target(param_name, self._param_list):
                    continue
                grad = param.main_grad if hasattr(param, "main_grad") else param.grad
                if grad is None:
                    logger.info(f"grad is None: {param_name}")
                    continue
                grad_info = GradStatCsv.generate_csv_line(param_name, self._level_adp, grad, self._bounds)
                output_lines.append(grad_info)
                if self._level_adp["have_grad_direction"]:
                    GradientMonitor.save_grad_direction(param_name, grad,
                                                        f'{self._output_path}/rank{self._rank}/step{self._step}')
            output_dirpath = os.path.join(self._output_path, f"rank{getattr(self, '_rank')}")
            if not os.path.isdir(output_dirpath):
                create_directory(output_dirpath)
            output_path = os.path.join(output_dirpath, f"grad_summary_{self._step}.csv")
            if os.path.exists(output_path):
                logger.warning(f"{output_path} will be deleted")
                remove_path(output_path)
            header_result = GradStatCsv.generate_csv_header(self._level_adp, self._bounds)
            output_lines.insert(0, header_result)
            write_csv(output_lines, output_path)
            logger.info(f"write grad data to {output_path}")

        if int(torch.__version__.split('.')[0]) >= 2:
            register_optimizer_step_pre_hook(optimizer_pre_step_hook)
