#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2022-2023. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""

import torch

from api_accuracy_checker.hook_module.hook_module import HOOKModule
from api_accuracy_checker.common.utils import torch_device_guard, torch_without_guard_version
from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.hook_module.utils import WrapNPUOps
from api_accuracy_checker.common.function_factory import npu_custom_functions

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False


def get_npu_ops():
    global WrapNPUOps
    if torch_without_guard_version:
        _npu_ops = dir(torch.ops.npu)
    else:
        _npu_ops = dir(torch_npu._C._VariableFunctionsClass)

    if msCheckerConfig.white_list:
        return set(WrapNPUOps) & set(_npu_ops) & set(msCheckerConfig.white_list)
    else:
        return set(WrapNPUOps) & set(_npu_ops)


class HOOKNpuOP(object):
    pass


class NPUOPTemplate(HOOKModule):

    def __init__(self, op_name, hook, need_hook=True):
        self.op_name_ = op_name
        self.prefix_op_name_ = "NPU*" + str(op_name) + "*"
        self.need_hook = need_hook
        if need_hook:
            super().__init__(hook)

    @torch_device_guard
    def forward(self, *args, **kwargs):
        if not self.need_hook:
            if self.op_name_ not in npu_custom_functions:
                raise Exception(f'There is not bench function {self.op_name_}')
            return npu_custom_functions[self.op_name_](*args, **kwargs)
        if torch_without_guard_version:
            return getattr(torch.ops.npu, str(self.op_name_))(*args, **kwargs)
        else:
            return getattr(torch_npu._C._VariableFunctionsClass, str(self.op_name_))(*args, **kwargs)


def wrap_npu_op(op_name, hook):

    def npu_op_template(*args, **kwargs):
        return NPUOPTemplate(op_name, hook)(*args, **kwargs)

    return npu_op_template


def wrap_npu_ops_and_bind(hook):
    _npu_ops = get_npu_ops()
    for op_name in _npu_ops:
        setattr(HOOKNpuOP, "wrap_" + str(op_name), wrap_npu_op(op_name, hook))