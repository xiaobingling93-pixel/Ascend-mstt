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
from msprobe.pytorch.function_factory import npu_custom_grad_functions


cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
ops = load_yaml(yaml_path)
wrap_aten_ops = ops.get('aten')
white_aten_ops = ops.get('white_aten_ops', [])


aten_func = {}
for f in dir(torch.ops.aten):
    aten_func[f] = getattr(torch.ops.aten, f)


def get_aten_ops():
    global wrap_aten_ops
    _all_aten_ops = dir(torch.ops.aten)
    return set(wrap_aten_ops) & set(_all_aten_ops)


class HOOKAtenOP(object):
    pass


class AtenOPTemplate(HOOKModule):
    def __init__(self, op, hook, need_hook=True):
        if isinstance(op, torch._ops.OpOverloadPacket):
            op_name_ = op._qualified_op_name.split("::")[-1]
        elif isinstance(op, str):
            op_name_ = str(op)
        else:
            op_name_ = op.name().split("::")[-1]
            overload_name = op._overloadname
            if not '.' + overload_name in op_name_:
                op_name_ = op_name_ + '.' + overload_name
        self.op = op
        self.prefix_op_name_ = "Aten" + Const.SEP + str(op_name_) + Const.SEP
        self.need_hook = need_hook
        if self.need_hook:
            super().__init__(hook)

    @torch_device_guard
    def forward(self, *args, **kwargs):
        if isinstance(self.op, str):
            if self.op in npu_custom_grad_functions:
                return npu_custom_grad_functions[self.op](*args, **kwargs)
            if self.op in white_aten_ops:
                return eval(f"torch.ops.aten.{self.op}")(*args, **kwargs)
            if self.op not in aten_func:
                raise Exception(f"Skip op[{self.op}] accuracy check, because the op is not "
                                f"in dir(torch.ops.aten) and support yaml.")
            return aten_func[self.op](*args, **kwargs)
        return self.op(*args, **kwargs)


class AtenOPPacketTemplate():
    def __init__(self, op_packet, hook):
        self.op_packet = op_packet
        self.hook = hook

    def __getattr__(self, key):
        try:
            attr = getattr(self.op_packet, key)
        except AttributeError as e:
            raise AttributeError(f"AtenOPPacketTemplate or OpOverloadPacket does not have attribute '{key}'.") from e
        if isinstance(attr, torch._ops.OpOverload):
            return AtenOPTemplate(attr, self.hook)
        else:
            return attr

    @torch_device_guard
    def __call__(self, *args, **kwargs):
        return AtenOPTemplate(self.op_packet, self.hook)(*args, **kwargs)

    def overloads(self):
        return self.op_packet.overloads()


def wrap_aten_op(op, hook):
    return AtenOPPacketTemplate(op, hook)


def wrap_aten_ops_and_bind(hook):
    _aten_ops = get_aten_ops()
    for op_name in _aten_ops:
        if not isinstance(aten_func.get(op_name), torch._ops.OpOverloadPacket):
            continue
        setattr(HOOKAtenOP, "wrap_" + str(op_name), wrap_aten_op(aten_func.get(op_name), hook))
