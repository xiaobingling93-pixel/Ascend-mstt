# Copyright 2024 Huawei Technologies Co., Ltd
#
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
# ============================================================================

from mindspore import Tensor, ops, mint
from mindspore.mint.nn import functional
from mindspore.common._stub_tensor import StubTensor
from mindspore.communication import comm_func

from msprobe.mindspore.dump.hook_cell.wrap_api import (HOOKTensor, HOOKStubTensor, HOOKFunctionalOP,
                                                       HOOKMintOP, HOOKMintNNFunctionalOP, HOOKDistributedOP,
                                                       get_wrap_api_list, setup_hooks)
from msprobe.core.common.utils import Const


class ApiRegistry:
    def __init__(self):
        self.tensor_ori_attr = {}
        self.stub_tensor_ori_attr = {}
        self.functional_ori_attr = {}
        self.mint_ops_ori_attr = {}
        self.mint_func_ops_ori_attr = {}
        self.distributed_ori_attr = {}
        self.norm_inner_ops_ori_attr = {}

        self.tensor_hook_attr = {}
        self.stub_tensor_hook_attr = {}
        self.functional_hook_attr = {}
        self.mint_ops_hook_attr = {}
        self.mint_func_ops_hook_attr = {}
        self.distibuted_hook_attr = {}
        self.norm_inner_ops_hook_attr = {}

        self.norm_inner_ops = ["norm", "square", "sqrt", "is_complex"]

    @staticmethod
    def store_ori_attr(ori_api_group, api_list, api_ori_attr):
        for api in api_list:
            if Const.SEP in api:
                sub_module_name, sub_op = api.rsplit(Const.SEP, 1)
                sub_module = getattr(ori_api_group, sub_module_name)
                api_ori_attr[api] = getattr(sub_module, sub_op)
            else:
                api_ori_attr[api] = getattr(ori_api_group, api)

    @staticmethod
    def set_api_attr(api_group, attr_dict):
        for api, api_attr in attr_dict.items():
            if Const.SEP in api:
                sub_module_name, sub_op = api.rsplit(Const.SEP, 1)
                sub_module = getattr(api_group, sub_module_name, None)
                if sub_module is not None:
                    setattr(sub_module, sub_op, api_attr)
            else:
                setattr(api_group, api, api_attr)

    def norm_inner_op_set_hook_func(self):
        self.set_api_attr(ops, self.norm_inner_ops_hook_attr)

    def norm_inner_op_set_ori_func(self):
        self.set_api_attr(ops, self.norm_inner_ops_ori_attr)

    def api_set_hook_func(self):
        self.set_api_attr(Tensor, self.tensor_hook_attr)
        self.set_api_attr(StubTensor, self.stub_tensor_hook_attr)
        self.set_api_attr(ops, self.functional_hook_attr)
        self.set_api_attr(mint, self.mint_ops_hook_attr)
        self.set_api_attr(functional, self.mint_func_ops_hook_attr)
        self.set_api_attr(comm_func, self.distibuted_hook_attr)

    def api_set_ori_func(self):
        self.set_api_attr(Tensor, self.tensor_ori_attr)
        self.set_api_attr(StubTensor, self.stub_tensor_ori_attr)
        self.set_api_attr(ops, self.functional_ori_attr)
        self.set_api_attr(mint, self.mint_ops_ori_attr)
        self.set_api_attr(functional, self.mint_func_ops_ori_attr)
        self.set_api_attr(comm_func, self.distributed_ori_attr)

    def initialize_hook(self, hook):
        wrap_api_name = get_wrap_api_list()
        self.store_ori_attr(Tensor, wrap_api_name.tensor_api_names, self.tensor_ori_attr)
        self.store_ori_attr(StubTensor, wrap_api_name.stub_tensor_api_names, self.stub_tensor_ori_attr)
        self.store_ori_attr(ops, wrap_api_name.ops_api_names, self.functional_ori_attr)
        self.store_ori_attr(mint, wrap_api_name.mint_api_names, self.mint_ops_ori_attr)
        self.store_ori_attr(functional, wrap_api_name.mint_nn_func_api_names, self.mint_func_ops_ori_attr)
        self.store_ori_attr(comm_func, wrap_api_name.distributed_api_names, self.distributed_ori_attr)
        self.store_ori_attr(ops, self.norm_inner_ops, self.norm_inner_ops_ori_attr)
        setup_hooks(hook)
        for attr_name in dir(HOOKTensor):
            if attr_name.startswith(Const.ATTR_NAME_PREFIX):
                api_name = attr_name[Const.ATTR_NAME_PREFIX_LEN:]
                self.tensor_hook_attr[api_name] = getattr(HOOKTensor, attr_name)
        for attr_name in dir(HOOKStubTensor):
            if attr_name.startswith(Const.ATTR_NAME_PREFIX):
                api_name = attr_name[Const.ATTR_NAME_PREFIX_LEN:]
                self.stub_tensor_hook_attr[api_name] = getattr(HOOKStubTensor, attr_name)
        for attr_name in dir(HOOKFunctionalOP):
            if attr_name.startswith(Const.ATTR_NAME_PREFIX):
                api_name = attr_name[Const.ATTR_NAME_PREFIX_LEN:]
                self.functional_hook_attr[api_name] = getattr(HOOKFunctionalOP, attr_name)
                if api_name in self.norm_inner_ops:
                    self.norm_inner_ops_hook_attr[api_name] = getattr(HOOKFunctionalOP, attr_name)
        for attr_name in dir(HOOKMintOP):
            if attr_name.startswith(Const.ATTR_NAME_PREFIX):
                api_name = attr_name[Const.ATTR_NAME_PREFIX_LEN:]
                self.mint_ops_hook_attr[api_name] = getattr(HOOKMintOP, attr_name)
        for attr_name in dir(HOOKMintNNFunctionalOP):
            if attr_name.startswith(Const.ATTR_NAME_PREFIX):
                api_name = attr_name[Const.ATTR_NAME_PREFIX_LEN:]
                self.mint_func_ops_hook_attr[api_name] = getattr(HOOKMintNNFunctionalOP, attr_name)
        for attr_name in dir(HOOKDistributedOP):
            if attr_name.startswith(Const.ATTR_NAME_PREFIX):
                api_name = attr_name[Const.ATTR_NAME_PREFIX_LEN:]
                self.distibuted_hook_attr[api_name] = getattr(HOOKDistributedOP, attr_name)


api_register = ApiRegistry()
