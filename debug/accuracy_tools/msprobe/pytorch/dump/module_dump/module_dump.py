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

import torch
from msprobe.core.common.const import Const
from msprobe.core.data_dump.scope import BaseScope
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.hook_module.api_registry import api_register

torch_version_above_or_equal_2 = torch.__version__.split('+')[0] >= '2.0'


class ModuleDumper:
    def __init__(self, service):
        self.service = service
        self.hook_handle_list = []

    def start_module_dump(self, module, dump_name):
        api_register.api_originality()
        self.register_hook(module, dump_name)

    def stop_module_dump(self):
        api_register.api_modularity()
        for hook_handle in self.hook_handle_list:
            if isinstance(hook_handle, torch.utils.hooks.RemovableHandle):
                hook_handle.remove()
        self.hook_handle_list.clear()

    def register_hook(self, module, dump_name):
        prefix_name = (
                BaseScope.Module_Type_Module + Const.SEP +
                dump_name + Const.SEP +
                module.__class__.__name__ + Const.SEP
        )
        module_processor = self.service.module_processor
        _, forward_hook, backward_hook, forward_hook_torch_version_below_2 = self.service.build_hook(
            BaseScope.Module_Type_Module,
            prefix_name
        )

        if module_processor.has_register_backward_hook(module):
            logger.warning(
                f"The {dump_name} module has registered deprecated register_backward_hook,"
                f"which may cause abnormal data dump. The backward data dump for this module will be skipped."
            )
        if torch_version_above_or_equal_2:
            forward_hook_handle = module.register_forward_hook(forward_hook, with_kwargs=True)
        else:
            if not module_processor.has_register_backward_hook(module):
                backward_hook_handle = module.register_full_backward_hook(
                    module_processor.node_hook(prefix_name + Const.BACKWARD, Const.STOP)
                )
                self.hook_handle_list.append(backward_hook_handle)
            forward_hook_handle = module.register_forward_hook(forward_hook_torch_version_below_2)
        self.hook_handle_list.append(forward_hook_handle)
        if not module_processor.has_register_backward_hook(module):
            backward_hook_handle = module.register_full_backward_hook(backward_hook)
            self.hook_handle_list.append(backward_hook_handle)

        forward_pre_hook_handle = module.register_forward_pre_hook(
            module_processor.node_hook(prefix_name + Const.FORWARD, Const.START)
        )
        forward_hook_handle = module.register_forward_hook(
            module_processor.node_hook(prefix_name + Const.FORWARD, Const.STOP)
        )
        self.hook_handle_list.extend([forward_pre_hook_handle, forward_hook_handle])
        if torch_version_above_or_equal_2 and not module_processor.has_register_backward_hook(module):
            backward_pre_hook_handle = module.register_full_backward_pre_hook(
                module_processor.node_hook(prefix_name + Const.BACKWARD, Const.START)
            )
            backward_hook_handle = module.register_full_backward_hook(
                module_processor.node_hook(prefix_name + Const.BACKWARD, Const.STOP)
            )
            self.hook_handle_list.extend([backward_pre_hook_handle, backward_hook_handle])
