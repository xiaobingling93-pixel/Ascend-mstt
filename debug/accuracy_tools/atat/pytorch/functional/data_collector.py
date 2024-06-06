
import os
from ..module_processer import ModuleProcesser
from .scope import BaseScope, build_scope, ListScope
from .json_writer import DataWriter
from ..common.log import print_info_log, print_info_log_rank_0, print_error_log_rank_0
from ..common.utils import Const
from ..common.file_check import FileOpen
from .data_processor import build_data_processor, DataProcessor


def build_collect_data(config):
    return DataCollector(config)


class DataCollector:
    overflow_task = "overflow_check"
    tensor_task = "tensor"
    freebenchmark_task = "free_benchmark"
    tasks_need_tensor_data = [overflow_task, tensor_task, freebenchmark_task]
    level_without_construct = "L1"

    def __init__(self, config):
        self.config = config
        self.data_writer = DataWriter()
        self.data_processor = build_data_processor(config, self.data_writer)
        self.module_count = {}
        if config.task == DataCollector.freebenchmark_task:
            self.scope = build_scope(ListScope, self.config.scope, self.config.list)
        else:
            self.scope = build_scope(None, self.config.scope, self.config.list)
    
    def if_return_forward_new_output(self):
        return self.data_processor.if_return_forward_new_output()
    
    def get_forward_new_output(self):
        return self.data_processor.get_forward_new_output()

    @property
    def dump_data_dir(self):
        return self.data_writer.dump_tensor_data_dir

    @property
    def dump_file_path(self):
        return self.data_writer.dump_file_path

    def visit_and_clear_overflow_status(self, api_or_module_name):
        self.data_processor.visit_and_clear_overflow_status(api_or_module_name)

    def write_json(self):
        self.data_writer.write_json()


    def update_data(self, data_info, msg=''):
        if self.config.task == DataProcessor.overflow:
            if self.data_processor.has_overflow:
                self.data_writer.update_data(data_info)
                msg += "Overflow detected."
            else:
                msg += "No Overflow, OK."
        else:
            self.data_writer.update_data(data_info)
        return msg

    @staticmethod
    def check_scope_and_pid(scope, name, pid):
        return (not scope or scope.check(name)) and pid == os.getpid()

    @staticmethod
    def is_inplace(module):
        return getattr(module, "op_is_inplace", False)

    def pre_forward(self, name, module_type, module, pid, module_input_output):
        backward_name = name.replace("forward", "backward")
        if self.check_scope_and_pid(self.scope, backward_name, pid):
            self.data_processor.analyze_pre_forward(backward_name, module, module_input_output)
        if not self.is_inplace(module):
            return
        print_info_log(f"API {name} is inplace.")
        if self.check_scope_and_pid(self.scope, name, pid):
            data_info = self.data_processor.analyze_pre_forward_inplace(name, module_input_output)
            self.update_data(data_info)

    def __call__(self, name_template, module_type, module, pid, module_input_output):
        if module_type == BaseScope.Module_Type_Module:
            name = module.mindstudio_reserved_name
        else:
            name = name_template

        if self.config.level != DataCollector.level_without_construct:
            self.data_writer.update_construct({name: ModuleProcesser.api_parent_node})
            self.data_writer.update_construct(ModuleProcesser.module_node)
        if not self.check_scope_and_pid(self.scope, name, pid):
            return
        msg = f"Calibrator is collecting data on {name}. "
        if "forward" in name:
            if not self.is_inplace(module):
                data_info = self.data_processor.analyze_forward(name, module, module_input_output)
            else:
                data_info = self.data_processor.analyze_forward_inplace(name, module_input_output)
            self.data_writer.update_stack(self.data_processor.analyze_api_call_stack(name))
        else:
            data_info = self.data_processor.analyze_backward(name, module, module_input_output)
        if data_info:
            msg = self.update_data(data_info, msg)
        print_info_log(msg)
        self.data_writer.flush_data_when_buffer_is_full()

    def module_count_func(self, name, name_template):
        module_name = name.split(Const.SEP)[-3]
        if "forward" in name_template:
            if module_name not in self.module_count:
                self.module_count[module_name] = [0, [0]]
            else:
                if self.module_count[module_name][-1] and \
                        self.module_count[module_name][0] != self.module_count[module_name][-1][-1]:
                    self.module_count[module_name][-1].pop()
                self.module_count[module_name][0] += 1
                self.module_count[module_name][-1].append(self.module_count[module_name][0])
            index = self.module_count[module_name][0]
        else:
            backward_stack = self.module_count[module_name][-1] if module_name in self.module_count else []
            if not backward_stack:
                index = "abnormal"
            else:
                index = backward_stack.pop()
        return index

    def update_dump_paths(self, *args):
        self.data_writer.update_dump_paths(*args)
        self.data_writer.initialize_json_file(task=self.config.task, level=self.config.level)
    
    def update_iter(self, current_iter):
        self.data_processor.update_iter(current_iter)
