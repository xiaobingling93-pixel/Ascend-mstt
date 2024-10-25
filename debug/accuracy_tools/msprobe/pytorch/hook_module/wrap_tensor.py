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

import os

import torch

from msprobe.pytorch.hook_module.hook_module import HOOKModule
from msprobe.pytorch.common.utils import torch_device_guard, parameter_adapter
from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_yaml


cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")


def get_tensor_ops():
    _tensor_ops = dir(torch.Tensor)
    yaml_data = load_yaml(yaml_path)
    wrap_tensor_ops = yaml_data.get('tensor')
    return set(wrap_tensor_ops) & set(_tensor_ops)


TensorOps = {op: getattr(torch.Tensor, op) for op in get_tensor_ops()}


class HOOKTensor(object):
    pass


class TensorOPTemplate(HOOKModule):

    def __init__(self, op_name, hook, need_hook=True):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Tensor" + Const.SEP + str(op_name) + Const.SEP
        if need_hook:
            super().__init__(hook)

    @torch_device_guard
    @parameter_adapter
    def forward(self, *args, **kwargs):
        return TensorOps[str(self.op_name_)](*args, **kwargs)


def wrap_tensor_op(op_name, hook):

    def tensor_op_template(*args, **kwargs):
        return TensorOPTemplate(op_name, hook)(*args, **kwargs)

    return tensor_op_template


def wrap_tensor_ops_and_bind(hook):
    _tensor_ops = get_tensor_ops()
    for op_name in _tensor_ops:
        setattr(HOOKTensor, "wrap_" + str(op_name), wrap_tensor_op(op_name, hook))
