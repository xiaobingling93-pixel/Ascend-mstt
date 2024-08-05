from functools import wraps

import torch
from torch.utils.hooks import BackwardHook

from msprobe.core.common.const import Const
from msprobe.core.data_dump.scope import ModuleRangeScope


class ModuleProcesser:
    module_count = {}
    module_stack = []
    api_parent_node = ""
    module_node = {}

    def __init__(self, scope):
        if isinstance(scope, ModuleRangeScope):
            self.scope = scope
        else:
            self.scope = None
        BackwardHook.setup_input_hook = ModuleProcesser.clone_return_value(BackwardHook.setup_input_hook)
        BackwardHook.setup_output_hook = ModuleProcesser.clone_return_value(BackwardHook.setup_output_hook)
        BackwardHook.setup_output_hook = ModuleProcesser.filter_tensor_and_tuple(BackwardHook.setup_output_hook)

    @staticmethod
    def filter_tensor_and_tuple(func):
        @wraps(func)
        def wrap_by_filter_tensor_and_tuple(*args, **kwargs):
            # setup_output_hook传入非tensor数据，工具后续dump会报错，处理方式是非tensor数据不传入
            # setup_output_hook定义为setup_output_hook(self, args)，因此处理第二个位置参数，即*args[1]
            if not isinstance(args[1], (torch.Tensor, tuple)):
                return args[1]
            return func(*args, **kwargs)

        return wrap_by_filter_tensor_and_tuple

    @staticmethod
    def clone_return_value(func):
        @wraps(func)
        def clone_return_value_func(*args, **kwargs):
            result = func(*args, **kwargs)
            return ModuleProcesser.clone_if_tensor(result)

        return clone_return_value_func
    
    @staticmethod
    def clone_if_tensor(result):
        if isinstance(result, torch.Tensor):
            return result.clone()
        elif isinstance(result, tuple):
            return tuple(ModuleProcesser.clone_if_tensor(x) for x in result)
        elif isinstance(result, list):
            return list(ModuleProcesser.clone_if_tensor(x) for x in result)
        elif isinstance(result, dict):
            return {k: ModuleProcesser.clone_if_tensor(v) for k, v in result.items()}
        else:
            return result

    @staticmethod
    def module_count_func(module_name):
        if module_name not in ModuleProcesser.module_count:
            ModuleProcesser.module_count[module_name] = 0
        else:
            ModuleProcesser.module_count[module_name] += 1
        return ModuleProcesser.module_count[module_name]

    @classmethod
    def reset_module_stats(cls):
        cls.module_count = {}
        cls.module_stack = []
        cls.api_parent_node = ""
        cls.module_node = {}

    def node_hook(self, name_prefix, start_or_stop, **kwargs):

        def pre_hook(module, input, output=None):
            try:
                index = ModuleProcesser.module_count_func(name_prefix)
            except IndexError as e:
                index = None
                pass
            module.mindstudio_reserved_name = full_name = name_prefix + Const.SEP + str(index)
            if self.module_stack:
                ModuleProcesser.module_node[full_name] = self.module_stack[-1]
            else:
                ModuleProcesser.module_node[full_name] = None

            ModuleProcesser.module_stack.append(full_name)
            if self.module_stack:
                ModuleProcesser.api_parent_node = self.module_stack[-1]
            if self.scope:
                self.scope.begin_module(full_name)

        def end_hook(module, input, output=None):
            if self.module_stack:
                ModuleProcesser.module_stack.pop()
            if self.module_stack:
                ModuleProcesser.api_parent_node = self.module_stack[-1]
            else:
                ModuleProcesser.api_parent_node = None
            if self.scope:
                self.scope.end_module(module.mindstudio_reserved_name)

        if Const.START in start_or_stop:
            return pre_hook
        else:
            return end_hook
