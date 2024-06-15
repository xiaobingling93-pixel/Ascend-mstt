from functools import wraps
import torch
from torch.utils.hooks import BackwardHook
from .functional.scope import ModuleRangeScope
from .common.utils import Const


class ModuleProcesser:
    module_stack = []
    api_parent_node = ""
    module_node = {}
    current_module_name = ""

    def __init__(self, scope):
        if isinstance(scope, ModuleRangeScope):
            self.scope = scope
        else:
            self.scope = None
        BackwardHook.setup_input_hook = ModuleProcesser.clone_return_value(BackwardHook.setup_input_hook)
        BackwardHook.setup_output_hook = ModuleProcesser.clone_return_value(BackwardHook.setup_output_hook)
        self.module_count = {}

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

    def node_hook(self, name_prefix, start_or_stop, **kwargs):

        def pre_hook(module, input, output=None):
            try:
                index = self.module_count_func(name_prefix)
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

    def module_count_func(self, module_name):
        if module_name not in self.module_count:
            self.module_count[module_name] = 0
        else:
            self.module_count[module_name] += 1
        return self.module_count[module_name]
