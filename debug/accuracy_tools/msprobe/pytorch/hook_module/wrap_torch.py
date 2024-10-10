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
from msprobe.pytorch.common.utils import torch_device_guard
from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_yaml


cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")


def get_torch_ops():
    _torch_ops = []
    yaml_data = load_yaml(yaml_path)
    wrap_torch_ops = yaml_data.get('torch')
    for operation in wrap_torch_ops:
        if '.' in operation:
            operation_sub_module_name, operation_sub_op = operation.rsplit('.', 1)
            operation_sub_module = getattr(torch, operation_sub_module_name)
            if operation_sub_op in dir(operation_sub_module):
                _torch_ops.append(operation)
        else:
            if hasattr(torch, operation):
                _torch_ops.append(operation)
    return set(_torch_ops)


TorchOps = {}
for op in get_torch_ops():
    if '.' in op:
        sub_module_name, sub_op = op.rsplit('.', 1)
        sub_module = getattr(torch, sub_module_name)
        TorchOps[op] = getattr(sub_module, sub_op)
    else:
        TorchOps[op] = getattr(torch, op)



class HOOKTorchOP(object):
    pass


class TorchOPTemplate(HOOKModule):

    def __init__(self, op_name, hook, need_hook=True):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Torch" + Const.SEP + str(op_name) + Const.SEP
        if need_hook:
            super().__init__(hook)

    @torch_device_guard
    def forward(self, *args, **kwargs):
        return TorchOps[str(self.op_name_)](*args, **kwargs)


def wrap_torch_op(op_name, hook):

    def torch_op_template(*args, **kwargs):
        return TorchOPTemplate(op_name, hook)(*args, **kwargs)

    return torch_op_template


def wrap_torch_ops_and_bind(hook):
    _torch_ops = get_torch_ops()
    for op_name in _torch_ops:
        setattr(HOOKTorchOP, "wrap_" + op_name, wrap_torch_op(op_name, hook))
