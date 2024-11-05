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

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_yaml
from msprobe.pytorch.hook_module.hook_module import HOOKModule
from msprobe.pytorch.common.utils import torch_device_guard


cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")


def get_vf_ops():
    yaml_data = load_yaml(yaml_path)
    wrap_vf_ops = yaml_data.get('_VF')
    return wrap_vf_ops


class HOOKVfOP(object):
    pass


class VfOPTemplate(HOOKModule):
    def __init__(self, op_name, hook):
        self.op_name_ = op_name
        self.prefix_op_name_ = "VF" + Const.SEP + str(op_name) + Const.SEP
        super().__init__(hook)

    @torch_device_guard
    def forward(self, *args, **kwargs):
        return getattr(torch._C._VariableFunctionsClass, str(self.op_name_))(*args, **kwargs)


def wrap_vf_op(op_name, hook):
    def vf_op_template(*args, **kwargs):
        return VfOPTemplate(op_name, hook)(*args, **kwargs)

    return vf_op_template


def wrap_vf_ops_and_bind(hook):
    _vf_ops = get_vf_ops()
    for op_name in _vf_ops:
        setattr(HOOKVfOP, "wrap_" + op_name, wrap_vf_op(op_name, hook))
