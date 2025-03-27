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

import mindspore
import mindspore as ms
from mindspore.common.api import jit
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.nn.optim.optimizer import Optimizer
from msprobe.core.common.file_utils import remove_path, write_csv, create_directory
from msprobe.core.grad_probe.constant import GradConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.grad_probe.global_context import grad_context
from msprobe.mindspore.grad_probe.grad_analyzer import csv_generator
from msprobe.mindspore.grad_probe.grad_analyzer import grad_dump, get_rank_id, GradDumpConfig
from msprobe.mindspore.grad_probe.grad_stat_csv import GradStatCsv, CsvInput
from msprobe.mindspore.grad_probe.utils import save_grad_direction, get_adapted_level


class HookInput:
    '''
    HookInput is a class wrapping all the variables used for hooking optimizer
    '''

    def __init__(self, opt) -> None:
        self.func = opt.construct
        if hasattr(opt, "_parameters"):
            parameter_list = opt._parameters
        elif hasattr(opt, "parameters"):
            parameter_list = opt.parameters
        else:
            logger.error_log_with_exp("Given optimizer has no attributes: '_parameters' or 'parameters'. \
                                      Please check the type of the given optimizer.", ValueError)
        self.g_names = [param.name for param in parameter_list]
        self.param_list = grad_context.get_context(GradConst.PARAM_LIST)
        self.rank_id = get_rank_id()
        output_path = grad_context.get_context(GradConst.OUTPUT_PATH)
        time_stamp = grad_context.get_context(GradConst.TIME_STAMP)
        self.dump_dir = os.path.join(output_path, f"rank{self.rank_id}", f"Dump{time_stamp}")
        self.save_dir = os.path.join(output_path, f"rank{self.rank_id}")
        self.step_finish_flag = os.path.join(self.dump_dir, GradConst.STEP_FINISH)
        self.level = grad_context.get_context(GradConst.LEVEL)
        self.bounds = grad_context.get_context(GradConst.BOUNDS)
        self.mode = mindspore.get_context("mode")


def hook_graph_mode_optimizer(opt, hook_input):
    @jit
    def new_construct(self, gradients):
        for index, grad_value in enumerate(gradients):
            if hook_input.param_list and hook_input.g_names[index] not in hook_input.param_list:
                continue
            conf = GradDumpConfig(dump_dir=hook_input.dump_dir, g_name=hook_input.g_names[index],
                                  dump_step=self.dump_step, grad=grad_value, level=hook_input.level,
                                  bounds=hook_input.bounds)
            grad_dump(conf)
        ms.ops.TensorDump()(hook_input.step_finish_flag, self.dump_step)
        self.assignadd(self.dump_step, self.global_step_increase_tensor)
        out = hook_input.func(gradients)
        return out

    opt.dump_step = Parameter(initializer(0, [1], ms.int32), name="dump_step")
    opt.construct = new_construct.__get__(opt, type(opt))
    csv_generator.start()


def hook_pynative_optimizer(opt, hook_input):
    level_adapted = get_adapted_level(hook_input.level)

    def hook_fn(cell, input_data):
        gradients, = input_data
        cur_step = grad_context.get_context(GradConst.CURRENT_STEP)
        if grad_context.step_need_dump(cur_step) and grad_context.rank_need_dump(hook_input.rank_id):
            create_directory(hook_input.save_dir)
            output_lines = []
            for index, grad_value in enumerate(gradients):
                param_name = hook_input.g_names[index]
                if hook_input.param_list and param_name not in hook_input.param_list:
                    continue
                csv_input = CsvInput(param_name, grad_value, hook_input.bounds)
                grad_info = GradStatCsv.get_csv_line(level_adapted, csv_input)
                output_lines.append(grad_info)
                if level_adapted["have_grad_direction"]:
                    save_grad_direction(param_name, grad_value, os.path.join(hook_input.save_dir, f'step{cur_step}'))
            output_csv_path = os.path.join(hook_input.save_dir, f"grad_summary_{cur_step}.csv")
            dummy_csv_input = CsvInput(None, None, hook_input.bounds)
            output_lines.insert(0, GradStatCsv.get_csv_header(level_adapted, dummy_csv_input))
            write_csv(output_lines, output_csv_path)
            logger.info(f"write grad data to {output_csv_path}")
        grad_context.update_step()

    opt.register_forward_pre_hook(hook_fn)


def hook_optimizer(opt: Optimizer):
    hook_input = HookInput(opt)

    if hook_input.mode == mindspore.GRAPH_MODE:
        hook_graph_mode_optimizer(opt, hook_input)
    else:
        hook_pynative_optimizer(opt, hook_input)
