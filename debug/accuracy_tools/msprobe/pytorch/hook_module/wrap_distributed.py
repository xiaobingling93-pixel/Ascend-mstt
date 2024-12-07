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
from functools import wraps
import torch.distributed as dist

from msprobe.pytorch.hook_module.hook_module import HOOKModule
from msprobe.pytorch.common.utils import torch_device_guard
from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_yaml
from msprobe.core.common.inplace_op_checker import InplaceOpChecker


cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")


distributed_func = {}
for f in dir(dist):
    distributed_func[f] = getattr(dist, f)


def get_distributed_ops():
    _all_distributed_ops = dir(dist)
    yaml_data = load_yaml(yaml_path)
    wrap_distributed_ops = yaml_data.get('distributed')
    return set(wrap_distributed_ops) & set(_all_distributed_ops)


class HOOKDistributedOP(object):
    pass


class DistributedOPTemplate(HOOKModule):
    def __init__(self, op_name, build_hook):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Distributed" + Const.SEP + str(op_name) + Const.SEP
        super().__init__(build_hook)
        if not self.stop_hook and InplaceOpChecker.check(self.op_name_, InplaceOpChecker.OP_DISTRIBUTED):
            self.op_is_inplace = True

    @torch_device_guard
    def forward(self, *args, **kwargs):
        if kwargs.get("async_op") or self.op_name_ in ["isend", "irecv"]:
            handle = distributed_func.get(self.op_name_)(*args, **kwargs)
            handle.wait()
            return handle
        else:
            return distributed_func.get(self.op_name_)(*args, **kwargs)


def wrap_distributed_op(op_name, hook):
    @wraps(DistributedOPTemplate)
    def distributed_op_template(*args, **kwargs):
        return DistributedOPTemplate(op_name, hook)(*args, **kwargs)

    distributed_op_template.__name__ = op_name
    return distributed_op_template


def wrap_distributed_ops_and_bind(hook):
    _distributed_ops = get_distributed_ops()
    for op_name in _distributed_ops:
        setattr(HOOKDistributedOP, "wrap_" + str(op_name), wrap_distributed_op(op_name, hook))
