#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import re
import torch

try:
    import torch_npu
except ImportError:
    current_device = "cuda"
else:
    current_device = "npu"

from msprobe.core.common.const import FileCheckConst, Const, CompareConst
from msprobe.core.common.file_utils import FileChecker
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException
from msprobe.pytorch.hook_module.wrap_aten import AtenOPTemplate
from msprobe.pytorch.hook_module.wrap_functional import FunctionalOPTemplate
from msprobe.pytorch.hook_module.wrap_npu_custom import NpuOPTemplate
from msprobe.pytorch.hook_module.wrap_tensor import TensorOPTemplate
from msprobe.pytorch.hook_module.wrap_torch import TorchOPTemplate

hf_32_standard_api = ["conv1d", "conv2d"]
not_detach_set = {'resize_', 'resize_as_', 'set_', 'transpose_', 't_', 'squeeze_', 'unsqueeze_'}
not_raise_dtype_set = {'type_as'}

PRECISION_MAPPING = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float64
}


class BackwardMessage:
    MULTIPLE_BACKWARD_MESSAGE = "Multiple backward is not supported."
    UNSUPPORT_BACKWARD_MESSAGE = "function with out=... arguments don't support automatic differentiation, " \
                                  "skip backward."
    NO_BACKWARD_RESULT_MESSAGE = "function backward result is None, skip backward."


class UtDataInfo:
    def __init__(self, bench_grad, device_grad, device_output, bench_output, grad_in, in_fwd_data_list,
                 backward_message, rank=0):
        self.bench_grad = bench_grad
        self.device_grad = device_grad
        self.device_output = device_output
        self.bench_output = bench_output
        self.grad_in = grad_in
        self.in_fwd_data_list = in_fwd_data_list
        self.backward_message = backward_message
        self.rank = rank


def get_validated_result_csv_path(result_csv_path, mode):
    if mode not in ['result', 'detail']:
        raise ValueError("The csv mode must be result or detail")
    result_csv_path_checker = FileChecker(result_csv_path, FileCheckConst.FILE, ability=FileCheckConst.READ_WRITE_ABLE,
                                          file_type=FileCheckConst.CSV_SUFFIX)
    validated_result_csv_path = result_csv_path_checker.common_check()
    if mode == 'result':
        result_csv_name = os.path.basename(validated_result_csv_path)
        pattern = r"^accuracy_checking_result_\d{14}\.csv$"
        if not re.match(pattern, result_csv_name):
            raise ValueError("When continue run ut, please do not modify the result csv name.")
    return validated_result_csv_path


def get_validated_details_csv_path(validated_result_csv_path):
    result_csv_name = os.path.basename(validated_result_csv_path)
    details_csv_name = result_csv_name.replace('result', 'details')
    details_csv_path = os.path.join(os.path.dirname(validated_result_csv_path), details_csv_name)
    details_csv_path_checker = FileChecker(details_csv_path, FileCheckConst.FILE,
                                           ability=FileCheckConst.READ_WRITE_ABLE, file_type=FileCheckConst.CSV_SUFFIX)
    validated_details_csv_path = details_csv_path_checker.common_check()
    return validated_details_csv_path


def exec_api(api_type, api_name, device, args, kwargs):
    if api_type == "Functional":
        torch_api = FunctionalOPTemplate(api_name, str, False)
    if api_type == "Tensor":
        torch_api = TensorOPTemplate(api_name, str, False)
    if api_type == "Torch":
        torch_api = TorchOPTemplate(api_name, str, False)
    if api_type == "Aten":
        torch_api = AtenOPTemplate(api_name, None, False)
    if api_type == "NPU":
        torch_api = NpuOPTemplate(api_name, None, False, device)
    out = torch_api.forward(*args, **kwargs)
    return out


def deal_detach(arg, to_detach=True):
    return arg.detach() if to_detach else arg


def raise_bench_data_dtype(api_name, arg, raise_dtype=None):
    '''
    将标杆数据的dtype转换为raise_dtype
    输入：
        api_name：api名称
        arg：标杆输入
        raise_dtype：需要转换的dtype
    输出： 
        arg: 转换dtype的标杆输入
    '''
    if api_name in hf_32_standard_api and arg.dtype == torch.float32:
        return arg
    if raise_dtype is None or arg.dtype not in PRECISION_MAPPING or raise_dtype == arg.dtype:
        return arg
    return arg.type(raise_dtype)


def generate_device_params(input_args, input_kwargs, need_backward, api_name):
    def recursive_arg_to_device(arg_in, to_detach, depth=0):
        if depth > Const.MAX_DEPTH:
            logger.error("The depth of arg_in is too large, please check the arg_in.")
            raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_device(arg, to_detach, depth=depth+1) for arg in arg_in)
        elif isinstance(arg_in, torch.Tensor):
            if need_backward and arg_in.requires_grad:
                arg_in = deal_detach(arg_in.clone(), to_detach).to(current_device).requires_grad_()
                temp_arg_in = arg_in * 1
                arg_in = temp_arg_in.type_as(arg_in)
                arg_in.retain_grad()
                return arg_in
            else:
                return deal_detach(arg_in.clone(), to_detach).to(current_device)
        else:
            return arg_in

    is_detach = api_name not in not_detach_set
    device_args = recursive_arg_to_device(input_args, is_detach)
    device_kwargs = \
        {key: recursive_arg_to_device(value, key != "out" and is_detach) for key, value in input_kwargs.items()}
    return device_args, device_kwargs


def generate_cpu_params(input_args, input_kwargs, need_backward, api_name):
    def recursive_arg_to_cpu(arg_in, to_detach, raise_dtype=None, depth=0):
        if depth > Const.MAX_DEPTH:
            logger.error("The depth of arg_in is too large, please check the arg_in.")
            raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_cpu(arg, to_detach, raise_dtype=raise_dtype, depth=depth+1) 
                                for arg in arg_in)
        elif isinstance(arg_in, torch.Tensor):
            if need_backward and arg_in.requires_grad:
                arg_in = deal_detach(raise_bench_data_dtype(
                                     api_name, arg_in.clone(), raise_dtype=raise_dtype), to_detach).requires_grad_()
                temp_arg_in = arg_in * 1
                arg_in = temp_arg_in.type_as(arg_in)
                arg_in.retain_grad()
                return arg_in
            else:
                return deal_detach(raise_bench_data_dtype(api_name, arg_in.clone(), raise_dtype=raise_dtype), to_detach)
        else:
            return arg_in

    def is_tensor_with_raise_precision(arg_in, check_kwargs=False):
        if arg_in.dtype in PRECISION_MAPPING:
            return True
        if check_kwargs and arg_in.dtype in [torch.half, torch.bfloat16]:
            return True
        return False

    def recursive_find_dtypes(arg_in, kwargs=None, check_kwargs=False, depth=0):
        if depth > Const.MAX_DEPTH:
            logger.error("The depth of arg_in is too large, please check the arg_in.")
            raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
        if isinstance(arg_in, (list, tuple)):
            return set().union(*tuple(recursive_find_dtypes(arg, kwargs, check_kwargs=check_kwargs, depth=depth+1) for arg in arg_in))
        elif isinstance(arg_in, torch.Tensor) and is_tensor_with_raise_precision(arg_in, check_kwargs):
            return set([arg_in.dtype])
        elif isinstance(arg_in, dict) and check_kwargs:
            return set().union(*tuple(recursive_find_dtypes(v, kwargs, check_kwargs=True, depth=depth+1) for v in arg_in.values()))
        return set()

    raise_dtype = None
    need_raise_dtypes = recursive_find_dtypes(input_args)
    need_raise_dtypes.update(recursive_find_dtypes(input_kwargs, check_kwargs=True))
    if len(need_raise_dtypes) == 1:
        raise_dtype = PRECISION_MAPPING.get(need_raise_dtypes.pop(), torch.float32)
    elif len(need_raise_dtypes) >= 2:
        raise_dtype = torch.float32

    raise_dtype = None if api_name in not_raise_dtype_set else raise_dtype
    is_detach = api_name not in not_detach_set
    cpu_args = recursive_arg_to_cpu(input_args, is_detach, raise_dtype=raise_dtype)
    cpu_kwargs = {key: recursive_arg_to_cpu(value, key != "out" and is_detach, raise_dtype=raise_dtype) for key, value in input_kwargs.items()}
    return cpu_args, cpu_kwargs


def record_skip_info(api_full_name, compare, compare_alg_results):
    result_info = (api_full_name, CompareConst.SKIP, CompareConst.SKIP, [compare_alg_results], None, 0)
    compare.record_results(result_info)
