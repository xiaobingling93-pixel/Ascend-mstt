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

from .hook_module import HOOKModule
from ..common.utils import torch_device_guard, Const
from ..common.file_check import FileOpen


cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with FileOpen(yaml_path, 'r') as f:
    WrapAtenOps = yaml.safe_load(f).get('aten')


aten_func = {}
for f in dir(torch.ops.aten):
    aten_func[f] = getattr(torch.ops.aten, f)


def get_aten_ops():
    global WrapAtenOps
    _all_aten_ops = dir(torch.ops.aten)
    return set(WrapAtenOps) & set(_all_aten_ops)


class HOOKAtenOP(object):
    pass


class AtenOPTemplate(HOOKModule):
    def __init__(self, op, hook):
        if isinstance(op, torch._ops.OpOverloadPacket):
            op_name_ = op._qualified_op_name.split("::")[-1]
        else:
            op_name_ = op.name().split("::")[-1]
            overload_name = op._overloadname
            if not '.' + overload_name in op_name_:
                op_name_ = op_name_ + '.' + overload_name
        self.op = op
        self.prefix_op_name_ = "Aten" + Const.SEP + str(op_name_) + Const.SEP
        super().__init__(hook)

    @torch_device_guard
    def forward(self, *args, **kwargs):
        return self.op(*args, **kwargs)


class AtenOPPacketTemplate():
    def __init__(self, opPacket, hook):
        self.opPacket = opPacket
        self.hook = hook

    def __getattr__(self, key):
        try:
            attr = getattr(self.opPacket, key)
        except AttributeError as e:
            raise AttributeError(f"AtenOPPacketTemplate or OpOverloadPacket does not have attribute '{key}'.") from e
        if isinstance(attr, torch._ops.OpOverload):
            return AtenOPTemplate(attr, self.hook)
        else:
            return attr

    def overloads(self):
        return self.opPacket.overloads()

    @torch_device_guard
    def __call__(self, *args, **kwargs):
        return AtenOPTemplate(self.opPacket, self.hook)(*args, **kwargs)


def wrap_aten_op(op, hook):
    return AtenOPPacketTemplate(op, hook)


def wrap_aten_ops_and_bind(hook):
    _aten_ops = get_aten_ops()
    for op_name in _aten_ops:
        if not isinstance(aten_func.get(op_name), torch._ops.OpOverloadPacket):
            continue
        setattr(HOOKAtenOP, "wrap_" + str(op_name), wrap_aten_op(aten_func.get(op_name), hook))
