# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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
from msprobe.core.common.const import Const
from msprobe.core.data_dump.scope import BaseScope, ModuleRangeScope, MixRangeScope
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import replace_last_occurrence
from torch.utils.checkpoint import checkpoint as origin_checkpoint
from torch.utils.checkpoint import set_checkpoint_early_stop
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
        replace_checkpoint()

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
        elif type(result) is tuple:
            return tuple(ModuleProcesser.clone_if_tensor(x) for x in result)
        elif type(result) is list:
            return list(ModuleProcesser.clone_if_tensor(x) for x in result)
        elif type(result) is dict:
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
    def has_register_backward_hook(module):
        return hasattr(module, '_backward_hooks') and \
            len(module._backward_hooks) > 0 and \
            module._is_full_backward_hook is False

    @staticmethod
    def get_modules_and_names(models):
        modules_and_names_with_index = {}
        if isinstance(models, (list, tuple)):
            for index, model in enumerate(models):
                modules_and_names_with_index[str(index)] = model.named_modules()
        else:
            modules_and_names_with_index["-1"] = models.named_modules()
        return modules_and_names_with_index

    @classmethod
    def reset_module_stats(cls):
        cls.module_count = {}
        cls.module_stack = []
        cls.api_parent_node = ""
        cls.module_node = {}

    def register_module_hook(self, models, build_hook):
        logger.info_on_rank_0("The init dump is enabled, and the module dump function will not be available.")
        modules_and_names_with_index = self.get_modules_and_names(models)
        for index, modules_and_names in modules_and_names_with_index.items():
            model = models if index == "-1" else models[int(index)]
            for name, module in modules_and_names:
                if module == model:
                    continue
                module_index = (index + Const.SEP) if index != "-1" else ""
                prefix_name = (BaseScope.Module_Type_Module + Const.SEP + module_index +
                               name + Const.SEP + module.__class__.__name__ + Const.SEP)
                pre_forward_hook, forward_hook, backward_hook, forward_hook_torch_version_below_2 = build_hook(
                    BaseScope.Module_Type_Module,
                    prefix_name
                )

                if self.has_register_backward_hook(module):
                    logger.warning(
                        f"The {prefix_name[:-1]} has registered deprecated register_backward_hook,"
                        f"which may cause abnormal data dump. The backward data dump for this module will be skipped."
                    )
                if torch_version_above_or_equal_2:
                    module.register_forward_hook(forward_hook, with_kwargs=True)
                else:
                    if not self.has_register_backward_hook(module):
                        module.register_full_backward_hook(self.node_hook(prefix_name + Const.BACKWARD, Const.STOP))
                    module.register_forward_hook(forward_hook_torch_version_below_2)
                if not self.has_register_backward_hook(module):
                    module.register_full_backward_hook(backward_hook)

                module.register_forward_pre_hook(self.node_hook(prefix_name + Const.FORWARD, Const.START))
                module.register_forward_hook(self.node_hook(prefix_name + Const.FORWARD, Const.STOP))
                if torch_version_above_or_equal_2 and not self.has_register_backward_hook(module):
                    module.register_full_backward_pre_hook(self.node_hook(prefix_name + Const.BACKWARD, Const.START))
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
            forward_full_name = replace_last_occurrence(full_name, Const.BACKWARD, Const.FORWARD)
            ModuleProcesser.module_node[full_name] = replace_last_occurrence(
                ModuleProcesser.module_node.get(forward_full_name), Const.FORWARD, Const.BACKWARD)
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
