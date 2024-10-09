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

import torch
import torch.distributed as dist

from msprobe.pytorch.hook_module import wrap_torch, wrap_functional, wrap_tensor, wrap_vf, wrap_distributed, wrap_aten
from msprobe.pytorch.hook_module.wrap_aten import get_aten_ops
from msprobe.pytorch.hook_module.wrap_distributed import get_distributed_ops
from msprobe.pytorch.hook_module.wrap_functional import get_functional_ops
from msprobe.pytorch.hook_module.wrap_tensor import get_tensor_ops
from msprobe.pytorch.hook_module.wrap_torch import get_torch_ops
from msprobe.pytorch.hook_module.wrap_vf import get_vf_ops
from msprobe.pytorch.common.utils import torch_without_guard_version, npu_distributed_api, is_gpu
from msprobe.core.common.const import Const

torch_version_above_2 = torch.__version__.split('+')[0] > '2.0'

if not is_gpu:
    import torch_npu
    from . import wrap_npu_custom
    from .wrap_npu_custom import get_npu_ops


class ApiRegistry:
    def __init__(self):
        self.tensor_ori_attr = {}
        self.torch_ori_attr = {}
        self.functional_ori_attr = {}
        self.distributed_ori_attr = {}
        self.npu_distributed_ori_attr = {}
        self.vf_ori_attr = {}
        self.aten_ori_attr = {}
        self.torch_npu_ori_attr = {}

        self.tensor_hook_attr = {}
        self.torch_hook_attr = {}
        self.functional_hook_attr = {}
        self.distributed_hook_attr = {}
        self.npu_distributed_hook_attr = {}
        self.vf_hook_attr = {}
        self.aten_hook_attr = {}
        self.torch_npu_hook_attr = {}

    @staticmethod
    def store_ori_attr(ori_api_group, api_list, api_ori_attr):
        for api in api_list:
            if '.' in api:
                sub_module_name, sub_op = api.rsplit('.', 1)
                sub_module = getattr(ori_api_group, sub_module_name)
                api_ori_attr[api] = getattr(sub_module, sub_op)
            else:
                api_ori_attr[api] = getattr(ori_api_group, api)

    @staticmethod
    def set_api_attr(api_group, attr_dict):
        for api, api_attr in attr_dict.items():
            if '.' in api:
                sub_module_name, sub_op = api.rsplit('.', 1)
                sub_module = getattr(api_group, sub_module_name, None)
                if sub_module is not None:
                    setattr(sub_module, sub_op, api_attr)
            else:
                setattr(api_group, api, api_attr)

    def api_modularity(self):
        self.set_api_attr(torch.Tensor, self.tensor_hook_attr)
        self.set_api_attr(torch, self.torch_hook_attr)
        self.set_api_attr(torch.nn.functional, self.functional_hook_attr)
        self.set_api_attr(dist, self.distributed_hook_attr)
        self.set_api_attr(dist.distributed_c10d, self.distributed_hook_attr)
        if not is_gpu and not torch_without_guard_version:
            self.set_api_attr(torch_npu.distributed, self.npu_distributed_hook_attr)
            self.set_api_attr(torch_npu.distributed.distributed_c10d, self.npu_distributed_hook_attr)
        if torch_version_above_2:
            self.set_api_attr(torch.ops.aten, self.aten_hook_attr)
        self.set_api_attr(torch._VF, self.vf_hook_attr)
        if not is_gpu:
            self.set_api_attr(torch_npu, self.torch_npu_hook_attr)

    def api_originality(self):
        self.set_api_attr(torch.Tensor, self.tensor_ori_attr)
        self.set_api_attr(torch, self.torch_ori_attr)
        self.set_api_attr(torch.nn.functional, self.functional_ori_attr)
        self.set_api_attr(dist, self.distributed_ori_attr)
        self.set_api_attr(dist.distributed_c10d, self.distributed_ori_attr)
        if not is_gpu and not torch_without_guard_version:
            self.set_api_attr(torch_npu.distributed, self.npu_distributed_ori_attr)
            self.set_api_attr(torch_npu.distributed.distributed_c10d, self.npu_distributed_ori_attr)
        if torch_version_above_2:
            self.set_api_attr(torch.ops.aten, self.aten_ori_attr)
        self.set_api_attr(torch._VF, self.vf_ori_attr)
        if not is_gpu:
            self.set_api_attr(torch_npu, self.torch_npu_ori_attr)

    def initialize_hook(self, hook, online_run_ut=False):
        """
        initialize_hook
        Args:
            hook (_type_): initialize_hook
            online_run_ut (bool): default False, whether online run_ut or not.
                If online_run_ut is True, the hook will not wrap the aten ops.
        """
        self.store_ori_attr(torch.Tensor, get_tensor_ops(), self.tensor_ori_attr)
        wrap_tensor.wrap_tensor_ops_and_bind(hook)
        for attr_name in dir(wrap_tensor.HOOKTensor):
            if attr_name.startswith(Const.ATTR_NAME_PREFIX):
                self.tensor_hook_attr[attr_name[5:]] = getattr(wrap_tensor.HOOKTensor, attr_name)

        self.store_ori_attr(torch, get_torch_ops(), self.torch_ori_attr)
        wrap_torch.wrap_torch_ops_and_bind(hook)
        for attr_name in dir(wrap_torch.HOOKTorchOP):
            if attr_name.startswith(Const.ATTR_NAME_PREFIX):
                self.torch_hook_attr[attr_name[5:]] = getattr(wrap_torch.HOOKTorchOP, attr_name)

        self.store_ori_attr(torch.nn.functional, get_functional_ops(), self.functional_ori_attr)
        wrap_functional.wrap_functional_ops_and_bind(hook)
        for attr_name in dir(wrap_functional.HOOKFunctionalOP):
            if attr_name.startswith(Const.ATTR_NAME_PREFIX):
                self.functional_hook_attr[attr_name[5:]] = getattr(wrap_functional.HOOKFunctionalOP, attr_name)

        self.store_ori_attr(dist, get_distributed_ops(), self.distributed_ori_attr)
        wrap_distributed.wrap_distributed_ops_and_bind(hook)
        if not is_gpu and not torch_without_guard_version:
            self.store_ori_attr(torch_npu.distributed, npu_distributed_api, self.npu_distributed_ori_attr)
        for attr_name in dir(wrap_distributed.HOOKDistributedOP):
            if attr_name.startswith(Const.ATTR_NAME_PREFIX):
                self.distributed_hook_attr[attr_name[5:]] = getattr(wrap_distributed.HOOKDistributedOP, attr_name)
                if not is_gpu and not torch_without_guard_version and attr_name[5:] in npu_distributed_api:
                    self.npu_distributed_hook_attr[attr_name[5:]] = getattr(wrap_distributed.HOOKDistributedOP,
                                                                            attr_name)

        if torch_version_above_2 and not online_run_ut:
            self.store_ori_attr(torch.ops.aten, get_aten_ops(), self.aten_ori_attr)
            wrap_aten.wrap_aten_ops_and_bind(hook)
            for attr_name in dir(wrap_aten.HOOKAtenOP):
                if attr_name.startswith(Const.ATTR_NAME_PREFIX):
                    self.aten_hook_attr[attr_name[5:]] = getattr(wrap_aten.HOOKAtenOP, attr_name)

        self.store_ori_attr(torch._VF, get_vf_ops(), self.vf_ori_attr)
        wrap_vf.wrap_vf_ops_and_bind(hook)
        for attr_name in dir(wrap_vf.HOOKVfOP):
            if attr_name.startswith(Const.ATTR_NAME_PREFIX):
                self.vf_hook_attr[attr_name[5:]] = getattr(wrap_vf.HOOKVfOP, attr_name)

        if not is_gpu:
            self.store_ori_attr(torch_npu, get_npu_ops(), self.torch_npu_ori_attr)
            wrap_npu_custom.wrap_npu_ops_and_bind(hook)
            for attr_name in dir(wrap_npu_custom.HOOKNpuOP):
                if attr_name.startswith(Const.ATTR_NAME_PREFIX):
                    self.torch_npu_hook_attr[attr_name[5:]] = getattr(wrap_npu_custom.HOOKNpuOP, attr_name)


api_register = ApiRegistry()
