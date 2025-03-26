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
from msprobe.core.common.exceptions import FreeBenchmarkException
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.pytorch.free_benchmark.common.enums import DeviceType


class Tools:

    @staticmethod
    def is_float_tensor(tensor) -> bool:
        if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
            return True
        if isinstance(tensor, (list, tuple)):
            for value in tensor:
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                    return True
        return False

    @staticmethod
    def get_dist_rank():
        try:
            return torch.distributed.get_rank()
        except RuntimeError:
            return 0

    @staticmethod
    def get_first_tensor_dtype(tensor_seq):
        if isinstance(tensor_seq, torch.Tensor):
            return tensor_seq.dtype
        if isinstance(tensor_seq, (list, tuple)):
            for object_ in tensor_seq:
                if isinstance(object_, torch.Tensor):
                    return object_.dtype
        raise RuntimeError("The sequence does not contain tensors.")

    @staticmethod
    def get_pure_api_name(api_name: str):
        return api_name.rsplit(".", 2)[0]

    @staticmethod
    @recursion_depth_decorator("FreeBenchmark: Tools.convert_device_and_dtype")
    def convert_device_and_dtype(
        tensor_seq, device: str = DeviceType.CPU, change_dtype: bool = False
    ):
        if isinstance(tensor_seq, torch.Tensor):
            if change_dtype and tensor_seq.dtype in [torch.float16, torch.bfloat16]:
                return tensor_seq.detach().to(device).to(torch.float32)
            return tensor_seq.detach().to(device)
        if isinstance(tensor_seq, dict):
            return {
                key: Tools.convert_device_and_dtype(value, device, change_dtype)
                for key, value in tensor_seq.items()
            }
        if isinstance(tensor_seq, (tuple, list)):
            return type(tensor_seq)(
                [
                    Tools.convert_device_and_dtype(value, device, change_dtype)
                    for value in tensor_seq
                ]
            )
        return tensor_seq

    @staticmethod
    @recursion_depth_decorator("FreeBenchmark: Tools.convert_fuzz_output_to_origin")
    def convert_fuzz_output_to_origin(origin, perturbed):
        if isinstance(origin, torch.Tensor) and isinstance(perturbed, torch.Tensor):
            origin.data = perturbed.to(origin.dtype).to(origin.device)
            return origin
        if isinstance(origin, dict) and isinstance(perturbed, dict):
            output = dict()
            for key, value in origin.items():
                if key not in perturbed:
                    err_msg = f"'{key}' not in perturbed output."
                    raise FreeBenchmarkException(
                        FreeBenchmarkException.InvalidPerturbedOutput,
                        error_info=err_msg,
                    )
                output[key] = Tools.convert_fuzz_output_to_origin(value, perturbed[key])
            return output
        if isinstance(origin, (tuple, list)) and isinstance(perturbed, (tuple, list)):
            result = list()
            if len(perturbed) != len(origin):
                err_msg = (
                    f"length of perturbed output ({len(perturbed)}) is different "
                    f"from the length of original output ({len(origin)})."
                )
                raise FreeBenchmarkException(
                    FreeBenchmarkException.InvalidPerturbedOutput, error_info=err_msg
                )
            for index_, value in enumerate(origin):
                result.append(
                    Tools.convert_fuzz_output_to_origin(value, perturbed[index_])
                )
            return type(origin)(result)
        err_msg = f"conversion of two outputs with types ({type(origin)}, {type(perturbed)}) is not supported."
        raise FreeBenchmarkException(
            FreeBenchmarkException.UnsupportedType, error_info=err_msg
        )


class TorchC:
    sum = torch._C._VariableFunctionsClass.sum
    isinf = torch._C._VariableFunctionsClass.isinf
    isfinite = torch._C._VariableFunctionsClass.isfinite
    isnan = torch._C._VariableFunctionsClass.isnan
    logical_not = torch._C._VariableFunctionsClass.logical_not
    subtract = torch._C._VariableFunctionsClass.subtract
    abs = torch._C._VariableFunctionsClass.abs
    where = torch._C._VariableFunctionsClass.where
    div = torch._C._VariableFunctionsClass.div
    mul = torch._C._VariableFunctionsClass.mul
    max = torch._C._VariableFunctionsClass.max
    min = torch._C._VariableFunctionsClass.min
    gt = torch._C._VariableFunctionsClass.gt
    ge = torch._C._VariableFunctionsClass.ge
    lt = torch._C._VariableFunctionsClass.lt
    mean = torch._C._VariableFunctionsClass.mean
    full = torch._C._VariableFunctionsClass.full
    add = torch._C._VariableFunctionsClass.add
    bitwise_xor = torch._C._VariableFunctionsClass.bitwise_xor
    clone = torch._C._VariableFunctionsClass.clone
    clamp = torch._C._VariableFunctionsClass.clamp
    tensor_split = torch._C._VariableFunctionsClass.tensor_split
    stack = torch._C._VariableFunctionsClass.stack
    reshape = torch._C._VariableFunctionsClass.reshape
    nan_to_num = torch._C._VariableFunctionsClass.nan_to_num
    aminmax = torch._C._VariableFunctionsClass.aminmax
