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
from api_accuracy_checker.hook_module.utils import WrapTensorOps
from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileOpen


def get_tensor_ops():
    global WrapTensorOps
    _tensor_ops = dir(torch._C._TensorBase)
    if msCheckerConfig.white_list:
        return set(WrapTensorOps) & set(_tensor_ops) & set(msCheckerConfig.white_list)
    else:
        return set(WrapTensorOps) & set(_tensor_ops)


class HOOKTensor(object):
    pass


class TensorOPTemplate(HOOKModule):

    def __init__(self, op_name, hook, need_hook=True):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Tensor*" + str(op_name) + "*"
        if need_hook:
            super().__init__(hook)

    @torch_device_guard
    def forward(self, *args, **kwargs):
        return getattr(torch._C._TensorBase, str(self.op_name_))(*args, **kwargs)


def wrap_tensor_op(op_name, hook):

    def tensor_op_template(*args, **kwargs):
        return TensorOPTemplate(op_name, hook)(*args, **kwargs)

    return tensor_op_template


def wrap_tensor_ops_and_bind(hook):
    _tensor_ops = get_tensor_ops()
    for op_name in _tensor_ops:
        setattr(HOOKTensor, "wrap_" + str(op_name), wrap_tensor_op(op_name, hook))
