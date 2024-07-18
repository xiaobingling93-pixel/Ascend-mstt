import os
from pathlib import Path
import functools
import mindspore

from atat.core.data_dump.data_collector import build_data_collector
from atat.core.data_dump.scope import BaseScope
from atat.mindspore.common.utils import get_rank_if_initialized
from atat.core.common.file_check import FileChecker, FileCheckConst, check_path_before_create
from atat.mindspore.common.log import logger
from atat.core.common.utils import Const
from atat.core.common.exceptions import DistributedNotInitializedError
from atat.mindspore.dump.hook_cell.api_registry import api_register
from atat.core.data_dump.data_processor.base import ModuleForwardInputsOutputs, ModuleBackwardInputsOutputs


class Service:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.config.level = self.config.level_ori
        self.data_collector = build_data_collector(config)
        self.switch = False
        self.current_iter = 0
        self.first_start = True
        self.current_rank = None
        self.dump_iter_dir = None

    def build_hook(self, module_type, name):
        def forward_hook(api_or_module_name, module, input, output):
            self.data_collector.visit_and_clear_overflow_status(api_or_module_name)
            if not self.switch:
                return
            if self.data_collector:
                module_input_output = ModuleForwardInputsOutputs(args=input, kwargs=module.input_kwargs, output=output)
                self.data_collector.forward_data_collect(api_or_module_name, module, pid, module_input_output)
                if self.data_collector.if_return_forward_new_output():
                    return self.data_collector.get_forward_new_output()
            return output

        def backward_hook(api_or_module_name, module, grad_input, grad_output):
            self.data_collector.visit_and_clear_overflow_status(api_or_module_name)
            if not self.switch:
                return
            if self.data_collector:
                module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_input, grad_output=grad_output)
                self.data_collector.backward_data_collect(api_or_module_name, module, pid, module_input_output)

        pid = os.getpid()
        forward_name_template = name + Const.FORWARD
        backward_name_template = name + Const.BACKWARD
        forward_hook = functools.partial(forward_hook, forward_name_template)
        backward_hook = functools.partial(backward_hook, backward_name_template)

        def wrap_forward_hook(*args, **kwargs):
            return forward_hook(*args, **kwargs)
        
        def wrap_backward_hook(*args, **kwargs):
            return backward_hook(*args, **kwargs)
        
        return wrap_forward_hook, wrap_backward_hook

    def step(self):
        self.current_iter += 1
        self.data_collector.update_iter(self.current_iter)

    def start(self, model):
        self.model = model
        if self.config.step and self.current_iter > max(self.config.step):
            self.stop()
            raise Exception("atat: exit after iteration {}".format(max(self.config.step)))
        if self.config.step and self.current_iter not in self.config.step:
            return
        if self.first_start:
            try:
                self.current_rank = get_rank_if_initialized()
            except DistributedNotInitializedError:
                self.current_rank = None

            if self.config.rank and self.current_rank not in self.config.rank:
                return
            self.register_hook_new()
            self.first_start = False
        self.switch = True
        logger.info_on_rank_0(f"Dump switch is turned on at step {self.current_iter}. ")
        if self.config.level != "L2":
            self.create_dirs()
            logger.info_on_rank_0(f"Dump data will be saved in {self.dump_iter_dir}.")

    def stop(self):
        if self.config.level == "L2":
            return
        if self.config.step and self.current_iter not in self.config.step:
            return
        if self.config.rank and self.current_rank not in self.config.rank:
            return
        self.switch = False
        self.data_collector.write_json()

    def create_dirs(self):
        check_path_before_create(self.config.dump_path)
        if not os.path.exists(self.config.dump_path):
            Path(self.config.dump_path).mkdir(mode=0o750, exist_ok=True)
        file_check = FileChecker(self.config.dump_path, FileCheckConst.DIR)
        file_check.common_check()
        self.dump_iter_dir = os.path.join(self.config.dump_path, f"step{self.current_iter}")
        cur_rank = self.current_rank if self.current_rank is not None else ''
        dump_dir = os.path.join(self.dump_iter_dir, f"rank{cur_rank}")
        if not os.path.exists(dump_dir):
            Path(dump_dir).mkdir(mode=0o750, parents=True, exist_ok=True)
        if self.config.task in self.data_collector.tasks_need_tensor_data:
            dump_data_dir = os.path.join(dump_dir, "dump_tensor_data")
            Path(dump_data_dir).mkdir(mode=0o750, exist_ok=True)
        else:
            dump_data_dir = None

        dump_file_path = os.path.join(dump_dir, "dump.json")
        stack_file_path = os.path.join(dump_dir, "stack.json")
        construct_file_path = os.path.join(dump_dir, "construct.json")
        free_benchmark_file_path = os.path.join(self.config.dump_path, "free_benchmark.csv")
        self.data_collector.update_dump_paths(
            dump_file_path, stack_file_path, construct_file_path, dump_data_dir, free_benchmark_file_path)
        
        
        
    def register_hook_new(self):
        logger.info_on_rank_0("The {} hook function is successfully mounted to the model.".format(self.config.task))
        if self.config.level == "L1":
            api_register.initialize_hook(functools.partial(self.build_hook, BaseScope.Module_Type_API))
            api_register.api_modularity()


