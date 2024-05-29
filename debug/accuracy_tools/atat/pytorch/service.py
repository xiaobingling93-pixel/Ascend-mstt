import os
from pathlib import Path
import functools
import torch
from .functional import build_repair, build_collect_data, build_step_post_process
from .functional.scope import BaseScope
from .common.utils import get_rank_if_initialized, is_gpu, Const
from .common.file_check import FileChecker, FileCheckConst, check_path_before_create
from .common import print_info_log_rank_0
from .common.exceptions import MsaccException
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
        def pre_hook(repair, name_template, module, args, kwargs):
            if repair:
                args, kwargs = repair.convert(name_template, module_type, args, kwargs)
            return args, kwargs

        def forward_hook(repair, name_template, module, args, kwargs, output):
            nonlocal module_type, pid
            if not self.switch:
                return
            if self.collect_data:
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=output)
                self.collect_data(name_template, module_type, module, pid, module_input_output)
            if repair:
                output = repair.invert(name_template, module_type, output)

            return output

        def backward_hook(repair, name_template, module, grad_input, grad_output):
            nonlocal module_type, pid
            if not self.switch:
                return
            if self.collect_data:
                module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_input, grad_output=grad_output)
                self.collect_data(name_template, module_type, module, pid, module_input_output)

        pid = os.getpid()
        if module_type == BaseScope.Module_Type_Module:
            forward_name_template = name + Const.SEP + "{}" + Const.SEP + "forward"
            backward_name_template = name + Const.SEP + "{}" + Const.SEP + "backward"
        else:
            forward_name_template = name + "forward"
            backward_name_template = name + "backward"
        pre_forward_hook = functools.partial(pre_hook, self.repair, forward_name_template)
        forward_hook = functools.partial(forward_hook, self.repair, forward_name_template)
        backward_hook = functools.partial(backward_hook, None, backward_name_template)
        return pre_forward_hook, forward_hook, backward_hook

    def step(self):
        self.current_iter += 1
        if self.step_post_process:
            self.step_post_process()

    @staticmethod
    def check_model_valid(model):
        if isinstance(model, torch.nn.Module):
            return model
        raise MsaccException(MsaccException.INVALID_PARAM_ERROR, "model 参数必须是torch.nn.Module类型。")

    def start(self, model):
        if self.config.step and self.current_iter > max(self.config.step):
            self.stop()
            raise Exception("atat: exit after iteration {}".format(max(self.config.step)))
        if self.config.step and self.current_iter not in self.config.step:
            return
        self.model = self.check_model_valid(model)
        if self.first_start:
            cur_rank = get_rank_if_initialized()
            self.current_rank = cur_rank if cur_rank is not None else ''
            if self.config.rank and self.current_rank not in self.config.rank:
                return
            self.register_hook_new()
            self.first_start = Falseq
        self.switch = True
        self.create_dirs()
        print_info_log_rank_0(f"Dump switch is turned on at step {self.current_iter}. "
                              f"Dump data will be saved in {self.dump_iter_dir}.")

    def stop(self):
        self.switch = False
        self.collect_data.write_json()


    def create_dirs(self):
        check_path_before_create(self.config.dump_path)
        if not os.path.exists(self.config.dump_path):
            Path(self.config.dump_path).mkdir(mode=0o750, exist_ok=True)
        file_check = FileChecker(self.config.dump_path, FileCheckConst.DIR)
        file_check.common_check()
        self.dump_iter_dir = os.path.join(self.config.dump_path, f"step{self.current_iter}")
        dump_dir = os.path.join(self.dump_iter_dir, f"rank{self.current_rank}")
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
        self.collect_data.update_dump_paths(dump_file_path, stack_file_path, construct_file_path, dump_data_dir)

    def register_hook_new(self):
        hook_name = self.config.task

        if "overflow_check" in hook_name and not is_gpu:
            pass

        print_info_log_rank_0("The {} hook function is successfully mounted to the model.".format(hook_name))
        if self.config.level in ["L0", "mix"]:
            assert self.model is not None
            print_info_log_rank_0("The init dump mode is enabled, and the module dump function will not be available")
            if not isinstance(self.model, torch.nn.Module):
                raise MsaccException(MsaccException.INVALID_PARAM_ERROR,
                                          "The argument model must be an object of torch.nn.Module")
            for name, module in self.model.named_modules():
                if module == self.model:
                    continue
                prefix = BaseScope.Module_Type_Module + Const.SEP + name + Const.SEP +\
                         module.__class__.__name__ + Const.SEP

                pre_forward_hook, forward_hook, backward_hook = self.build_hook(BaseScope.Module_Type_Module, prefix)
                module.register_forward_hook(forward_hook, with_kwargs=True)
                module.register_full_backward_hook(backward_hook)

                module.register_forward_pre_hook(self.module_processor.node_hook(prefix + "forward", "start"))
                module.register_forward_hook(self.module_processor.node_hook(prefix + "forward", "stop"))
                module.register_full_backward_pre_hook(self.module_processor.node_hook(prefix + "backward", "start"))
                module.register_full_backward_hook(self.module_processor.node_hook(prefix + "backward", "stop"))

        if self.config.level in ["mix", "L1"]:
            api_register.initialize_hook(functools.partial(self.build_hook, BaseScope.Module_Type_API))
            api_register.api_modularity()

        if "acc_cmp_dump" in hook_name:
            remove_dropout()

