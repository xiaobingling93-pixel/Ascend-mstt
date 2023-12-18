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
import numpy as np

from api_accuracy_checker.common.utils import Const, check_file_or_directory_path, check_object_type, print_warn_log, print_error_log, \
    CompareException

TORCH_TYPE = ["torch.device", "torch.dtype"]
TENSOR_DATA_LIST = ["torch.Tensor", "torch.nn.parameter.Parameter"]
FLOAT_TYPE = ['torch.float32', 'torch.float', 'torch.float64', 'torch.double', 'torch.float16',
              'torch.half', 'torch.bfloat16']


def gen_data(info, need_grad, convert_type):
    """
    Function Description:
        Based on arg basic information, generate arg data
    Parameter:
        info: arg basic information. Dict
        need_grad: set Tensor grad for backward
        convert_type: convert ori_type to dist_type flag.
    """
    check_object_type(info, dict)
    data_type = info.get('type')
    data_path = info.get('datapath')
    if data_type in TENSOR_DATA_LIST:
        if data_path:
            data = gen_real_tensor(data_path, convert_type)
        else:
            data = gen_random_tensor(info, convert_type)
        if info.get('requires_grad') and need_grad:
            data.requires_grad_(True)
            temp_data = data * 1
            data = temp_data.type_as(data)
            data.retain_grad()
    else:
        data = info.get('value')
        if info.get("type") == "slice":
            data = slice(*data)
    return data


def gen_real_tensor(data_path, convert_type):
    """
    Function Description:
        Based on API data path, generate input parameters real data
    Parameter:
        data_path: API data path
        convert_type: convert ori_type to dist_type flag.
    """
    data_path = os.path.realpath(data_path)
    check_file_or_directory_path(data_path)
    if not data_path.endswith('.pt') and not data_path.endswith('.npy'):
        error_info = f"The file: {data_path} is not a pt or numpy file."
        raise CompareException(CompareException.INVALID_FILE_ERROR, error_info)
    if data_path.endswith('.pt'):
        data = torch.load(data_path)
    else:
        data_np = np.load(data_path)
        data = torch.from_numpy(data_np)
    if convert_type:
        ori_dtype = Const.CONVERT.get(convert_type)[0]
        dist_dtype = Const.CONVERT.get(convert_type)[1]
        if str(data.dtype) == ori_dtype:
            data = data.type(eval(dist_dtype))
    return data


def gen_random_tensor(info, convert_type):
    """
    Function Description:
        Based on API MAX and MIN, generate input parameters random data
    Parameter:
        info: API data info
        convert_type: convert ori_type to dist_type flag.
    """
    check_object_type(info, dict)
    low, high = info.get('Min'), info.get('Max')
    data_dtype = info.get('dtype')
    shape = tuple(info.get('shape'))
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        error_info = f'Data info Min: {low} , Max: {high}, info type must be int or float.'
        raise CompareException(CompareException.INVALID_PARAM_ERROR, error_info)
    if data_dtype == "torch.bool":
        data = gen_bool_tensor(low, high, shape)
    else:
        data = gen_common_tensor(low, high, shape, data_dtype, convert_type)
    return data


def gen_common_tensor(low, high, shape, data_dtype, convert_type):
    """
    Function Description:
        Based on API basic information, generate int or float tensor
    Parameter:
        low: The minimum value in Tensor
        high: The max value in Tensor
        shape:The shape of Tensor
        data_dtype: The data type of Tensor
        convert_type: convert ori_type to dist_type flag.
    """
    if convert_type:
        ori_dtype = Const.CONVERT.get(convert_type)[0]
        if ori_dtype == data_dtype:
            data_dtype = Const.CONVERT.get(convert_type)[1]
    if data_dtype in FLOAT_TYPE:
        if high in [float('inf'), float('-inf')] or low in [float('inf'), float('-inf')]:
            error_info = 'Parameter contains inf, skip comparison.'
            raise CompareException(CompareException.INVALID_PARAM_ERROR, error_info)
        scale = high - low
        rand01 = torch.rand(shape, dtype=eval(data_dtype))
        tensor = rand01 * scale + low
    elif 'int' in data_dtype or 'long' in data_dtype:
        low, high = int(low), int(high)
        tensor = torch.randint(low, high + 1, shape, dtype=eval(data_dtype))
    else:
        print_error_log('Dtype is not supported: ' + data_dtype)
        raise NotImplementedError()
    if tensor.nelement() == 0:
        return tensor
    tmp_tensor = tensor.reshape(-1)
    tmp_tensor[0] = low
    tmp_tensor[-1] = high
    data = tmp_tensor.reshape(shape)
    return data


def gen_bool_tensor(low, high, shape):
    """
    Function Description:
        Based on API basic information, generate bool tensor
    Parameter:
        low: The minimum value in Tensor
        high: The max value in Tensor
        shape:The shape of Tensor
    """
    low, high = int(low), int(high)
    tensor = torch.randint(low, high + 1, shape)
    data = torch.gt(tensor, 0)
    return data


def gen_args(args_info, need_grad=True, convert_type=None):
    """
    Function Description:
        Based on API basic information, generate input parameters: args, for API forward running
    Parameter:
        api_info: API basic information. List
        need_grad: set Tensor grad for backward
        convert_type: convert ori_type to dist_type flag.
    """
    check_object_type(args_info, list)
    args_result = []
    for arg in args_info:
        if isinstance(arg, (list, tuple)):
            data = gen_args(arg, need_grad, convert_type)
        elif isinstance(arg, dict):
            data = gen_data(arg, need_grad, convert_type)
        else:
            print_warn_log(f'Warning: {arg} is not supported')
            raise NotImplementedError()
        args_result.append(data)
    return args_result


def gen_kwargs(api_info, convert_type=None):
    """
    Function Description:
        Based on API basic information, generate input parameters: kwargs, for API forward running
    Parameter:
        api_info: API basic information. Dict
        convert_type: convert ori_type to dist_type flag.
    """
    check_object_type(api_info, dict)
    kwargs_params = api_info.get("kwargs")
    for key, value in kwargs_params.items():
        if isinstance(value, (list, tuple)):
            kwargs_params[key] = gen_list_kwargs(value, convert_type)
        elif value.get('type') in TENSOR_DATA_LIST:
            kwargs_params[key] = gen_data(value, False, convert_type)
        elif value.get('type') in TORCH_TYPE:
            gen_torch_kwargs(kwargs_params, key, value)
        else:
            kwargs_params[key] = value.get('value')
    return kwargs_params


def gen_torch_kwargs(kwargs_params, key, value):
    if value.get('type') == "torch.device":
        kwargs_params[key] = eval(value.get('type'))(value.get('value'))
    else:
        kwargs_params[key] = eval(value.get('value'))


def gen_list_kwargs(kwargs_item_value, convert_type):
    """
    Function Description:
        When kwargs value is list, generate the list of kwargs result
    Parameter:
        kwargs_item_value: kwargs value before to generate. List
        convert_type: convert ori_type to dist_type flag.
    """
    kwargs_item_result = []
    for item in kwargs_item_value:
        if item.get('type') in TENSOR_DATA_LIST:
            item_value = gen_data(item, False, convert_type)
        else:
            item_value = item.get('value')
        kwargs_item_result.append(item_value)
    return kwargs_item_result


def gen_api_params(api_info, need_grad=True, convert_type=None):
    """
    Function Description:
        Based on API basic information, generate input parameters: args, kwargs, for API forward running
    Parameter:
        api_info: API basic information. Dict
        need_grad: set grad for backward
        convert_type: convert ori_type to dist_type flag.
    """
    check_object_type(api_info, dict)
    if convert_type and convert_type not in Const.CONVERT:
        error_info = f"convert_type params not support {convert_type}."
        raise CompareException(CompareException.INVALID_PARAM_ERROR, error_info)
    kwargs_params = gen_kwargs(api_info, convert_type)
    if api_info.get("args"):
        args_params = gen_args(api_info.get("args"), need_grad, convert_type)
    else:
        print_warn_log(f'Warning: No args in {api_info} ')
        args_params = []
    return args_params, kwargs_params
