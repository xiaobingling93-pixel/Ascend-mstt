#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2023-2023. Huawei Technologies Co., Ltd. All rights reserved.
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

import os

import torch
import yaml

from api_accuracy_checker.hook_module.hook_module import HOOKModule
from api_accuracy_checker.common.utils import torch_device_guard
from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.hook_module.utils import WrapTorchOps
from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileOpen


def get_torch_ops():
    global WrapTorchOps
    _torch_ops = dir(torch._C._VariableFunctionsClass)
    if msCheckerConfig.white_list:
        return set(WrapTorchOps) & set(_torch_ops) & set(msCheckerConfig.white_list)
    else:
        return set(WrapTorchOps) & set(_torch_ops)


class HOOKTorchOP(object):
    pass


class TorchOPTemplate(HOOKModule):

    def __init__(self, op_name, hook, need_hook=True):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Torch*" + str(op_name) + "*"
        if need_hook:
            super().__init__(hook)

    def input_param_need_adapt(self):
        special_op_list = ["broadcast_tensors", "block_diag"]
        for item in special_op_list:
            if item in self.op_name_:
                return True
        return False

    def einsum_adapt(self, *args):
        if len(args) < 2:
            raise ValueError('einsum(): must specify the equation string and at least one operand, '
                             'or at least one operand and its subscripts list')
        equation = None
        operands = None
        if isinstance(args[0], torch.Tensor):
            def parse_subscript(n: int) -> str:
                if n == Ellipsis:
                    return '...'
                if n >= 0 and n < 26:
                    return chr(ord('A') + n)
                if n >= 26 and n < 52:
                    return chr(ord('a') + n - 26)
                raise ValueError('einsum(): subscript in subscript list is not within the valid range [0, 52]')
            equation = ','.join(''.join(parse_subscript(s) for s in l) for l in args[1::2])

            if len(args) % 2 == 1:
                equation += '->' + ''.join(parse_subscript(s) for s in args[-1])
                operands = args[:-1:2]
            else:
                operands = args[::2]
        else:
            equation = args[0]
            operands = args[1:]

        if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
            _operands = operands[0]
            return self.einsum_adapt(equation, *_operands)
        return equation, operands

    @torch_device_guard
    def forward(self, *args, **kwargs):
        if self.input_param_need_adapt():
            return getattr(torch._C._VariableFunctionsClass, str(self.op_name_))(args, **kwargs)
        else:
            if self.op_name_ == 'einsum':
                args = self.einsum_adapt(*args)
            return getattr(torch._C._VariableFunctionsClass, str(self.op_name_))(*args, **kwargs)


def wrap_torch_op(op_name, hook):

    def torch_op_template(*args, **kwargs):
        return TorchOPTemplate(op_name, hook)(*args, **kwargs)

    return torch_op_template


def wrap_torch_ops_and_bind(hook):
    _torch_ops = get_torch_ops()
    for op_name in _torch_ops:
        setattr(HOOKTorchOP, "wrap_" + op_name, wrap_torch_op(op_name, hook))
