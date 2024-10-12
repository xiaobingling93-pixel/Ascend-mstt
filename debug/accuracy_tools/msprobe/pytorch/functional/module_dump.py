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

import torch
import torch.nn as nn
from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.data_dump.scope import BaseScope
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.debugger.precision_debugger import PrecisionDebugger
from msprobe.pytorch.hook_module.api_registry import api_register
from msprobe.pytorch.service import torch_version_above_or_equal_2

hook_handle_list = []


def module_dump(module, dump_name):
    if not isinstance(module, nn.Module):
        logger.error("The parameter module in module_dump must be a Module subclass.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if not isinstance(dump_name, str):
        logger.error("The parameter dump_name in module_dump must be a str type.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)

    api_register.api_originality()
    register_hook(module, dump_name)


def module_dump_end():
    api_register.api_modularity()
    remove_hook()
    hook_handle_list.clear()


def register_hook(module, dump_name):
    prefix = BaseScope.Module_Type_Module + Const.SEP + dump_name + Const.SEP + module.__class__.__name__ + Const.SEP

    pdg = PrecisionDebugger()
    _, forward_hook, backward_hook, forward_hook_torch_version_below_2 = \
        pdg.service.build_hook(BaseScope.Module_Type_Module, prefix)

    if torch_version_above_or_equal_2:
        forward_hook_handle = module.register_forward_hook(forward_hook, with_kwargs=True)
        hook_handle_list.append(forward_hook_handle)
    else:
        pdg.service.check_register_full_backward_hook(module)
        full_backward_hook_handle = module.register_full_backward_hook(
            pdg.service.module_processor.node_hook(prefix + Const.BACKWARD, Const.STOP))
        forward_hook_handle = module.register_forward_hook(forward_hook_torch_version_below_2)
        hook_handle_list.extend([full_backward_hook_handle, forward_hook_handle])
    pdg.service.check_register_full_backward_hook(module)
    full_backward_hook_handle = module.register_full_backward_hook(backward_hook)

    forward_pre_hook_handle = module.register_forward_pre_hook(
        pdg.service.module_processor.node_hook(prefix + Const.FORWARD, Const.START))
    forward_hook_handle = module.register_forward_hook(
        pdg.service.module_processor.node_hook(prefix + Const.FORWARD, Const.STOP))
    hook_handle_list.extend([full_backward_hook_handle, forward_pre_hook_handle, forward_hook_handle])

    if torch_version_above_or_equal_2:
        backward_pre_hook_handle = module.register_full_backward_pre_hook(
            pdg.service.module_processor.node_hook(prefix + Const.BACKWARD, Const.START))
        pdg.service.check_register_full_backward_hook(module)
        full_backward_hook_handle = module.register_full_backward_hook(
            pdg.service.module_processor.node_hook(prefix + Const.BACKWARD, Const.STOP))
        hook_handle_list.extend([backward_pre_hook_handle, full_backward_hook_handle])


def remove_hook():
    for hook_handle in hook_handle_list:
        if isinstance(hook_handle, torch.utils.hooks.RemovableHandle):
            hook_handle.remove()
