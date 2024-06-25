import os
from pathlib import Path
import functools
import torch
from .functional import build_repair, build_collect_data, build_step_post_process
from .functional.scope import BaseScope
from .common.utils import get_rank_if_initialized, is_gpu, Const
from .common.file_check import FileChecker, FileCheckConst, check_path_before_create
from .common import print_info_log_rank_0
from .hook_module.api_registry import api_register
from .hook_module import remove_dropout
from .functional.data_processor import ModuleForwardInputsOutputs, ModuleBackwardInputsOutputs
from .module_processer import ModuleProcesser


class Service:
    make_dir_flag = True
    REGISTER_HOOK_KWARGS = ["overflow_nums", "dump_mode", "dump_config"]

    def __init__(self, config):
        self.model = None
        self.config = config
        self.collect_data = build_collect_data(config)
        self.module_processor = ModuleProcesser(self.collect_data.scope)
        self.repair = build_repair(config)
        self.step_post_process = build_step_post_process(config)
        self.switch = False
        self.current_iter = 0
        self.first_start = True
        self.current_rank = None
        self.first_touch_dir = True

    def build_hook(self, module_type, name):
        def pre_hook(repair, api_or_module_name, module, args, kwargs):
            nonlocal module_type, pid
            if module_type == BaseScope.Module_Type_Module:
                api_or_module_name = module.mindstudio_reserved_name
            self.collect_data.visit_and_clear_overflow_status(api_or_module_name)

            if not self.switch:
                return args, kwargs
            if repair:
                args, kwargs = repair.convert(api_or_module_name, module_type, args, kwargs)
            if self.collect_data:
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=None)
                self.collect_data.pre_forward(api_or_module_name, module_type, module, pid, module_input_output)
            return args, kwargs

        def forward_hook(repair, api_or_module_name, module, args, kwargs, output):
            nonlocal module_type, pid
            if module_type == BaseScope.Module_Type_Module:
                api_or_module_name = module.mindstudio_reserved_name
            self.collect_data.visit_and_clear_overflow_status(api_or_module_name)

            if not self.switch:
                return
            if self.collect_data:
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=output)
                self.collect_data(api_or_module_name, module_type, module, pid, module_input_output)
                if self.collect_data.if_return_forward_new_output():
                    return self.collect_data.get_forward_new_output()
            if repair:
                output = repair.invert(api_or_module_name, module_type, output)

            return output

        def backward_hook(repair, api_or_module_name, module, grad_input, grad_output):
            nonlocal module_type, pid
            if module_type == BaseScope.Module_Type_Module:
                api_or_module_name = module.mindstudio_reserved_name
            self.collect_data.visit_and_clear_overflow_status(api_or_module_name)

            if not self.switch:
                return
            if self.collect_data:
                module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_input, grad_output=grad_output)
                self.collect_data(api_or_module_name, module_type, module, pid, module_input_output)

        pid = os.getpid()
        forward_name_template = name + Const.FORWARD
        backward_name_template = name + Const.BACKWARD
        pre_forward_hook = functools.partial(pre_hook, self.repair, forward_name_template)
        forward_hook = functools.partial(forward_hook, self.repair, forward_name_template)
        backward_hook = functools.partial(backward_hook, None, backward_name_template)
        return pre_forward_hook, forward_hook, backward_hook

    def step(self):
        self.current_iter += 1
        if self.step_post_process:
            self.step_post_process()
        self.collect_data.update_iter(self.current_iter)

    def start(self, model):
        self.model = model
        if self.config.step and self.current_iter > max(self.config.step):
            self.stop()
            raise Exception("atat: exit after iteration {}".format(max(self.config.step)))
        if self.config.step and self.current_iter not in self.config.step:
            return
        if self.first_start:
            self.current_rank = get_rank_if_initialized()
            if self.config.rank and self.current_rank not in self.config.rank:
                return
            self.register_hook_new()
            self.first_start = False
        self.switch = True
        print_info_log_rank_0(f"Dump switch is turned on at step {self.current_iter}. ")
        if self.config.level != "L2":
            self.create_dirs()
            print_info_log_rank_0(f"Dump data will be saved in {self.dump_iter_dir}.")

    def stop(self):
        if self.config.level == "L2":
            return
        if self.config.step and self.current_iter not in self.config.step:
            return
        if self.config.rank and self.current_rank not in self.config.rank:
            return
        self.switch = False
        self.collect_data.write_json()


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
        if self.config.task in self.collect_data.tasks_need_tensor_data:
            dump_data_dir = os.path.join(dump_dir, "dump_tensor_data")
            Path(dump_data_dir).mkdir(mode=0o750, exist_ok=True)
        else:
            dump_data_dir = None

        dump_file_path = os.path.join(dump_dir, "dump.json")
        stack_file_path = os.path.join(dump_dir, "stack.json")
        construct_file_path = os.path.join(dump_dir, "construct.json")
        free_benchmark_file_path = os.path.join(self.config.dump_path, "free_benchmark.csv")
        self.collect_data.update_dump_paths(dump_file_path, stack_file_path, construct_file_path, dump_data_dir, free_benchmark_file_path)

    def register_hook_new(self):
        hook_name = self.config.task

        if "overflow_check" in hook_name and not is_gpu:
            pass

        print_info_log_rank_0("The {} hook function is successfully mounted to the model.".format(hook_name))
        if self.config.level in ["L0", "mix"]:
            assert self.model is not None
            print_info_log_rank_0("The init dump mode is enabled, and the module dump function will not be available")
            for name, module in self.model.named_modules():
                if module == self.model:
                    continue
                prefix = BaseScope.Module_Type_Module + Const.SEP + name + Const.SEP +\
                         module.__class__.__name__ + Const.SEP

                pre_forward_hook, forward_hook, backward_hook = self.build_hook(BaseScope.Module_Type_Module, prefix)
                module.register_forward_hook(forward_hook, with_kwargs=True)
                module.register_full_backward_hook(backward_hook)

                module.register_forward_pre_hook(
                    self.module_processor.node_hook(prefix + Const.FORWARD, Const.START))
                module.register_forward_hook(
                    self.module_processor.node_hook(prefix + Const.FORWARD, Const.STOP))
                module.register_full_backward_pre_hook(
                    self.module_processor.node_hook(prefix + Const.BACKWARD, Const.START))
                module.register_full_backward_hook(
                    self.module_processor.node_hook(prefix + Const.BACKWARD, Const.STOP))

        if self.config.level in ["mix", "L1", "L2"]:
            api_register.initialize_hook(functools.partial(self.build_hook, BaseScope.Module_Type_API))
            api_register.api_modularity()

        if Const.STATISTICS in hook_name or Const.TENSOR in hook_name:
            remove_dropout()

