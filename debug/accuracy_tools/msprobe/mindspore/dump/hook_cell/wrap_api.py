# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

from mindspore import Tensor, mint, ops
from mindspore.common._stub_tensor import StubTensor
from mindspore.communication import comm_func
from mindspore.mint.nn import functional

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_yaml
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.common.utils import is_mindtorch
from msprobe.mindspore.dump.hook_cell.hook_cell import HOOKCell

if is_mindtorch():
    import torch
    import torch_npu

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, MsConst.SUPPORTED_API_LIST_FILE)
torch_yaml_path = os.path.join(cur_path, "../../../pytorch/hook_module", MsConst.SUPPORTED_API_LIST_FILE)


class HOOKTensor(object):
    pass


class HOOKStubTensor(object):
    pass


class HOOKFunctionalOP(object):
    pass


class HOOKMintOP(object):
    pass


class HOOKMintNNFunctionalOP(object):
    pass


class HOOKDistributedOP(object):
    pass


class HOOKTorchOP(object):
    pass


class HOOKTorchTensor(object):
    pass


class HOOKTorchFunctionalOP(object):
    pass


class HOOKTorchDistributedOP(object):
    pass


class HOOKTorchNpuOP(object):
    pass


class ApiTemplate(HOOKCell):
    def __init__(self, api_name, api_dict, prefix, hook):
        self.api_name = api_name
        self.api_func = api_dict[api_name]
        self.prefix_api_name = prefix + str(api_name.split(Const.SEP)[-1]) + Const.SEP
        super().__init__(hook)

    @staticmethod
    def async_to_sync(output):
        # Fake handle, used to return after the CommHandle executes the wait method
        fake_handle = type("FakeHandle", (), {"wait": lambda self: None})()
        if isinstance(output, tuple) and len(output) == 2 and hasattr(output[1], "wait"):
            output[1].wait()
            output = (output[0], fake_handle)
        elif hasattr(output, "wait"):
            output.wait()
            output = fake_handle
        return output

    def construct(self, *args, **kwargs):
        if self.api_name.startswith(MsConst.DROPOUT_API_NAME_PREFIX):
            return args[0] if args else kwargs.get(Const.INPUT)
        
        output = self.api_func(*args, **kwargs)

        if self.prefix_api_name.startswith(MsConst.DISTRIBUTED_DATA_PREFIX):
            if kwargs.get("async_op") or self.api_name in ["isend", "irecv"]:
                output = self.async_to_sync(output)
        return output

    def forward(self, *args, **kwargs):
        if self.api_name.startswith(MsConst.DROPOUT_API_NAME_PREFIX):
            return args[0] if args else kwargs.get(Const.INPUT)
        return self.api_func(*args, **kwargs)


class WrapApiName:
    def __init__(self, tensor_api_names, stub_tensor_api_names, ops_api_names, mint_api_names, mint_nn_func_api_names,
                 distributed_api_names):
        self.tensor_api_names = tensor_api_names
        self.stub_tensor_api_names = stub_tensor_api_names
        self.ops_api_names = ops_api_names
        self.mint_api_names = mint_api_names
        self.mint_nn_func_api_names = mint_nn_func_api_names
        self.distributed_api_names = distributed_api_names


class WrapTorchApiName:
    def __init__(self, torch_api_names, tensor_api_names, functional_api_names, distributed_api_names, npu_api_names):
        self.torch_api_names = torch_api_names
        self.tensor_api_names = tensor_api_names
        self.functional_api_names = functional_api_names
        self.distributed_api_names = distributed_api_names
        self.npu_api_names = npu_api_names


def get_wrap_api_list():
    api_list = load_yaml(yaml_path)
    tensor_api = api_list.get(MsConst.SUPPORTED_TENSOR_LIST_KEY)
    ops_api = api_list.get(MsConst.SUPPORTED_OPS_LIST_KEY)
    mint_api = api_list.get(MsConst.SUPPORTED_MINT_LIST_KEY)
    mint_nn_func_api = api_list.get(MsConst.SUPPORTED__MINT_NN_FUNC_LIST_KEY)
    distributed_api = api_list.get(MsConst.SUPPORTED_COMM_LIST_KEY)
    wrap_api_name = WrapApiName(set(tensor_api) & set(dir(Tensor)),
                                set(tensor_api) & set(dir(StubTensor)),
                                set(ops_api) & set(dir(ops)),
                                set(mint_api) & set(dir(mint)),
                                set(mint_nn_func_api) & set(dir(functional)),
                                set(distributed_api) & set(dir(comm_func)))
    return wrap_api_name


def get_wrap_torch_api_list():
    api_list = load_yaml(torch_yaml_path)
    torch_api = api_list.get("torch")
    tensor_api = api_list.get("tensor")
    functional_api = api_list.get("functional")
    distributed_api = api_list.get("distributed")
    npu_api = api_list.get("torch_npu")
    wrap_api_name = WrapTorchApiName(set(torch_api) & set(dir(torch)),
                                     set(tensor_api) & set(dir(torch.Tensor)),
                                     set(functional_api) & set(dir(torch.nn.functional)),
                                     set(distributed_api) & set(dir(torch.distributed)),
                                     set(npu_api) & set(dir(torch_npu)))
    return wrap_api_name


def wrap_api_func(api_name, api_dict, prefix, hook):
    def api_function(*args, **kwargs):
        return ApiTemplate(api_name, api_dict, prefix, hook)(*args, **kwargs)
    return api_function


def wrap_api_func_and_bind(api_list, api_dict, prefix, hook, hook_class):
    for api_name in api_list:
        if callable(api_dict[api_name]):
            setattr(hook_class, Const.ATTR_NAME_PREFIX + api_name, wrap_api_func(api_name, api_dict, prefix, hook))


def setup_hooks(hook):
    if is_mindtorch():
        torch_wrap_api_name = get_wrap_torch_api_list()
        wrap_api_func_and_bind(torch_wrap_api_name.torch_api_names,
                               {f: getattr(torch, f) for f in dir(torch)},
                               MsConst.TORCH_DATA_PREFIX, hook, HOOKTorchOP)
        wrap_api_func_and_bind(torch_wrap_api_name.tensor_api_names,
                               {f: getattr(torch.Tensor, f) for f in dir(torch.Tensor)},
                               MsConst.TENSOR_DATA_PREFIX, hook, HOOKTorchTensor)
        wrap_api_func_and_bind(torch_wrap_api_name.functional_api_names,
                               {f: getattr(torch.nn.functional, f) for f in dir(torch.nn.functional)},
                               MsConst.OPS_DATA_PREFIX, hook, HOOKTorchFunctionalOP)
        wrap_api_func_and_bind(torch_wrap_api_name.distributed_api_names,
                               {f: getattr(torch.distributed, f) for f in dir(torch.distributed)},
                               MsConst.DISTRIBUTED_DATA_PREFIX, hook, HOOKTorchDistributedOP)
        wrap_api_func_and_bind(torch_wrap_api_name.npu_api_names, {f: getattr(torch_npu, f) for f in dir(torch_npu)},
                               MsConst.TORCH_NPU_DATA_PREFIX, hook, HOOKTorchNpuOP)
        return

    wrap_api_name = get_wrap_api_list()
    wrap_api_func_and_bind(wrap_api_name.tensor_api_names, {f: getattr(Tensor, f) for f in dir(Tensor)},
                           MsConst.TENSOR_DATA_PREFIX, hook, HOOKTensor)
    wrap_api_func_and_bind(wrap_api_name.stub_tensor_api_names, {f: getattr(StubTensor, f) for f in dir(StubTensor)},
                           MsConst.STUB_TENSOR_DATA_PREFIX, hook, HOOKStubTensor)
    wrap_api_func_and_bind(wrap_api_name.ops_api_names, {f: getattr(ops, f) for f in dir(ops)},
                           MsConst.OPS_DATA_PREFIX, hook, HOOKFunctionalOP)
    wrap_api_func_and_bind(wrap_api_name.mint_api_names, {f: getattr(mint, f) for f in dir(mint)},
                           MsConst.MINT_DATA_PREFIX, hook, HOOKMintOP)
    wrap_api_func_and_bind(wrap_api_name.mint_nn_func_api_names, {f: getattr(functional, f) for f in dir(functional)},
                           MsConst.MINT_NN_FUNC_DATA_PREFIX, hook, HOOKMintNNFunctionalOP)
    wrap_api_func_and_bind(wrap_api_name.distributed_api_names, {f: getattr(comm_func, f) for f in dir(comm_func)},
                           MsConst.DISTRIBUTED_DATA_PREFIX, hook, HOOKDistributedOP)
