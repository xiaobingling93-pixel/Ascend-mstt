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
from msprobe.pytorch.common.utils import torch_device_guard, torch_without_guard_version
from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import load_yaml
from msprobe.pytorch.function_factory import npu_custom_functions

try:
    import torch_npu
except ImportError:
    logger.info("Failing to import torch_npu.")


cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
cuda_func_mapping = {"npu_fusion_attention" : "gpu_fusion_attention"}


def get_npu_ops():
    if torch_without_guard_version:
        _npu_ops = dir(torch.ops.npu)
    else:
        _npu_ops = dir(torch_npu._C._VariableFunctionsClass)
    yaml_data = load_yaml(yaml_path)
    wrap_npu_ops = yaml_data.get('torch_npu')
    return set(wrap_npu_ops) & set(_npu_ops)


class HOOKNpuOP(object):
    pass


class NpuOPTemplate(HOOKModule):

    def __init__(self, op_name, hook, need_hook=True, device=Const.CPU_LOWERCASE):
        self.op_name_ = op_name
        self.prefix_op_name_ = "NPU" + Const.SEP + str(op_name) + Const.SEP
        self.need_hook = need_hook
        self.device = device
        if need_hook:
            super().__init__(hook)

    @torch_device_guard
    def forward(self, *args, **kwargs):
        if not self.need_hook:
            if self.op_name_ not in npu_custom_functions:
                raise Exception(f'There is not bench function {self.op_name_}')
            if self.device == Const.CUDA_LOWERCASE:
                self.op_name_ = cuda_func_mapping.get(self.op_name_, self.op_name_)
            if self.device in [Const.CUDA_LOWERCASE, Const.CPU_LOWERCASE]:
                return npu_custom_functions[self.op_name_](*args, **kwargs)
        if torch_without_guard_version:
            return getattr(torch.ops.npu, str(self.op_name_))(*args, **kwargs)
        else:
            return getattr(torch_npu._C._VariableFunctionsClass, str(self.op_name_))(*args, **kwargs)


def wrap_npu_op(op_name, hook):
    def npu_op_template(*args, **kwargs):
        return NpuOPTemplate(op_name, hook)(*args, **kwargs)
    return npu_op_template


def wrap_npu_ops_and_bind(hook):
    _npu_ops = get_npu_ops()
    for op_name in _npu_ops:
        setattr(HOOKNpuOP, "wrap_" + str(op_name), wrap_npu_op(op_name, hook))
