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

from functools import wraps

import torch
from torch.utils.checkpoint import set_checkpoint_early_stop
from torch.utils.checkpoint import checkpoint as origin_checkpoint
from msprobe.core.common.const import Const
from msprobe.core.data_dump.scope import BaseScope, ModuleRangeScope, MixRangeScope
from msprobe.pytorch.common.log import logger
from torch.utils.hooks import BackwardHook

torch_version_above_or_equal_2 = torch.__version__.split('+')[0] >= '2.0'

def checkpoint_without_early_stop(*args, **kwargs):
    with set_checkpoint_early_stop(False):
        return origin_checkpoint(*args, **kwargs)

def replace_checkpoint():
    torch.utils.checkpoint.checkpoint = checkpoint_without_early_stop

class ModuleProcesser:
    module_count = {}
    module_stack = []
    api_parent_node = ""
    module_node = {}

    def __init__(self, scope):
        self.scope = scope if isinstance(scope, (ModuleRangeScope, MixRangeScope)) else None
        BackwardHook.setup_input_hook = ModuleProcesser.clone_return_value(BackwardHook.setup_input_hook)
        BackwardHook.setup_output_hook = ModuleProcesser.clone_return_value(BackwardHook.setup_output_hook)
        BackwardHook.setup_output_hook = ModuleProcesser.filter_tensor_and_tuple(BackwardHook.setup_output_hook)
        replace_checkpoint()

    @staticmethod
    def filter_tensor_and_tuple(func):
        @wraps(func)
        def wrap_by_filter_tensor_and_tuple(*args, **kwargs):
            # setup_output_hook传入非tensor数据，工具后续dump会报错，处理方式是解析非tensor数据的属性，对tensor属性挂hook
            # setup_output_hook定义为setup_output_hook(self, args)，因此处理第二个位置参数，即*args[1]
            if not isinstance(args[1], (torch.Tensor, tuple)):
                for item_str in dir(args[1]):
                    item = getattr(args[1], item_str)
                    # 处理tensor或者只包含tensor的元组
                    if isinstance(item, torch.Tensor) or \
                            (isinstance(item, tuple) and all(isinstance(x, torch.Tensor) for x in item)):
                        args_new = (args[0], item)
                        result = func(*args_new, **kwargs)
                        setattr(args[1], item_str, result)
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

    @staticmethod
    def remove_deprecated_backward_hook_if_exist(module):
        if hasattr(module, '_backward_hooks') and \
                len(module._backward_hooks) > 0 and \
                module._is_full_backward_hook is False:
            module._backward_hooks.clear()
            module._is_full_backward_hook = None
            logger.warning("Found deprecated backward hooks. Removing them and switching to full backward hooks.")

    @classmethod
    def reset_module_stats(cls):
        cls.module_count = {}
        cls.module_stack = []
        cls.api_parent_node = ""
        cls.module_node = {}

    def hook_modules(self, models, build_hook):
        logger.info_on_rank_0("The init dump is enabled, and the module dump function will not be available.")
        for model in models:
            self.register_module_hook(model, build_hook)

    def register_module_hook(self, model, build_hook):
        for name, module in model.named_modules():
            if module == model:
                continue

            prefix_name = (
                    BaseScope.Module_Type_Module + Const.SEP +
                    name + Const.SEP +
                    module.__class__.__name__ + Const.SEP
            )
            pre_forward_hook, forward_hook, backward_hook, forward_hook_torch_version_below_2 = build_hook(
                BaseScope.Module_Type_Module,
                prefix_name
            )
            if torch_version_above_or_equal_2:
                module.register_forward_hook(forward_hook, with_kwargs=True)
            else:
                self.remove_deprecated_backward_hook_if_exist(module)
                module.register_full_backward_hook(self.node_hook(prefix_name + Const.BACKWARD, Const.STOP))
                module.register_forward_hook(forward_hook_torch_version_below_2)
            self.remove_deprecated_backward_hook_if_exist(module)
            module.register_full_backward_hook(backward_hook)

            module.register_forward_pre_hook(self.node_hook(prefix_name + Const.FORWARD, Const.START))
            module.register_forward_hook(self.node_hook(prefix_name + Const.FORWARD, Const.STOP))
            if torch_version_above_or_equal_2:
                module.register_full_backward_pre_hook(self.node_hook(prefix_name + Const.BACKWARD, Const.START))
                self.remove_deprecated_backward_hook_if_exist(module)
                module.register_full_backward_hook(self.node_hook(prefix_name + Const.BACKWARD, Const.STOP))

    def node_hook(self, name_prefix, start_or_stop, **kwargs):

        def pre_hook(module, input, output=None):
            try:
                index = ModuleProcesser.module_count_func(name_prefix)
            except IndexError as e:
                index = None
                pass
            full_name = name_prefix + Const.SEP + str(index)
            if not hasattr(module, "mindstudio_reserved_name") or not module.mindstudio_reserved_name:
                module.mindstudio_reserved_name = []
            module.mindstudio_reserved_name.append(full_name)
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
            if not hasattr(module, "mindstudio_reserved_name") or not module.mindstudio_reserved_name:
                raise RuntimeError(f"module reserve name is None when pop")
            current_name = module.mindstudio_reserved_name.pop()
            if self.scope:
                self.scope.end_module(current_name)

        def backward_hook(module, input, output=None):
            try:
                index = ModuleProcesser.module_count_func(name_prefix)
            except IndexError as e:
                index = None
                pass
            full_name = name_prefix + Const.SEP + str(index)
            if not hasattr(module, "mindstudio_reserved_name") or not module.mindstudio_reserved_name:
                module.mindstudio_reserved_name = []
            module.mindstudio_reserved_name.append(full_name)
            forward_full_name = full_name.replace(Const.BACKWARD, Const.FORWARD)
            ModuleProcesser.module_node[full_name] = ModuleProcesser.module_node[forward_full_name].replace(
                Const.FORWARD, Const.BACKWARD) if ModuleProcesser.module_node[forward_full_name] else None
            ModuleProcesser.api_parent_node = None
            if self.scope:
                self.scope.begin_module(full_name)

        if torch_version_above_or_equal_2:
            if Const.START in start_or_stop:
                return pre_hook
            else:
                return end_hook
        else:
            if Const.FORWARD in name_prefix and Const.START in start_or_stop:
                return pre_hook
            elif Const.BACKWARD in name_prefix:
                return backward_hook
            else:
                return end_hook
