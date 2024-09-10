
import os

import mindspore
import mindspore as ms
from mindspore.common.api import jit
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer

from msprobe.core.grad_probe.constant import GradConst
from msprobe.mindspore.common.log import logger

from msprobe.core.common.file_utils import remove_path, write_csv, create_directory
from msprobe.mindspore.grad_probe.global_context import grad_context
from msprobe.mindspore.grad_probe.grad_analyzer import grad_dump, get_rank_id
from msprobe.mindspore.grad_probe.grad_analyzer import csv_generator
from msprobe.mindspore.grad_probe.grad_stat_csv import GradStatCsv, CsvInput
from msprobe.mindspore.grad_probe.utils import save_grad_direction, get_adapted_level

class HookInput:

    '''
    HookInput is a class wrapping all the variables used for hooking optimizer
    '''

    def __init__(self, opt) -> None:
        self.func = opt.construct
        self.g_names = [param.name for param in opt._parameters]
        self.param_list = grad_context.get_context(GradConst.PARAM_LIST)
        self.rank_id = get_rank_id()
        output_path = grad_context.get_context(GradConst.OUTPUT_PATH)
        self.dump_dir = os.path.join(output_path, f"rank{self.rank_id}", "Dump")
        self.save_dir = os.path.join(output_path, f"rank{self.rank_id}")
        self.step_finish_flag = os.path.join(self.dump_dir, GradConst.STEP_FINISH)
        if os.path.exists(self.save_dir):
            logger.warning(f"Delete existing path {self.save_dir}.")
            remove_path(self.save_dir)
        self.level = grad_context.get_context(GradConst.LEVEL)
        self.bounds = grad_context.get_context(GradConst.BOUNDS)
        self.mode = mindspore.get_context("mode")

def hook_graph_mode_optimizer(opt, hook_input):
    @jit
    def new_construct(self, gradients):
        for index, grad_value in enumerate(gradients):
            if hook_input.param_list and hook_input.g_names[index] not in hook_input.param_list:
                continue
            grad_dump(hook_input.dump_dir, hook_input.g_names[index], self.dump_step,
                    grad_value, hook_input.level, hook_input.bounds)
        ms.ops.TensorDump()(hook_input.step_finish_flag, self.dump_step)
        self.assignadd(self.dump_step, self.global_step_increase_tensor)
        out = hook_input.func(gradients)
        return out

    opt.dump_step = Parameter(initializer(0, [1], ms.int32), name="dump_step")
    opt.construct = new_construct.__get__(opt, type(opt))
    csv_generator.start()

def hook_pynative_optimizer(opt, hook_input):
    level_adapted = get_adapted_level(hook_input.level)

    def hook_fn(cell, input):
        gradients, = input
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
