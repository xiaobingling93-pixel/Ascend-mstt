#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
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
import random
import zlib
from functools import wraps

import torch
import numpy as np

from atat.core.utils import print_error_log
from atat.core.utils import Const
from atat.core.utils import CompareException



try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False

torch_without_guard_version_list = ['2.1', '2.2']
for version in torch_without_guard_version_list:
    if torch.__version__.startswith(version):
        torch_without_guard_version = True
        break
    else:
        torch_without_guard_version = False

if not is_gpu and not torch_without_guard_version:
    from torch_npu.utils.device_guard import torch_device_guard as torch_npu_device_guard


def check_is_npu():
    return not is_gpu


def torch_device_guard(func):
    if is_gpu or torch_without_guard_version:
        return func
    # Parse args/kwargs matched torch.device objects

    @torch_npu_device_guard
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def seed_all(seed=1234, mode=False):
    check_seed_all(seed, mode)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)
    if is_gpu:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.benchmark = False
    else:
        torch_npu.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)


def check_seed_all(seed, mode):
    if isinstance(seed, int):
        if seed < 0 or seed > Const.MAX_SEED_VALUE:
            print_error_log(f"Seed must be between 0 and {Const.MAX_SEED_VALUE}.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
    else:
        print_error_log(f"Seed must be integer.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    if not isinstance(mode, bool):
        print_error_log(f"seed_all mode must be bool.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)


def parameter_adapter(func):

    @wraps(func)
    def inner(self, *args, **kwargs):
        if self.op_name_ == "__getitem__" and len(args) > 1 and isinstance(args[1], torch.Tensor):
            input_tensor = args[0]
            indices = args[1]
            if indices.dtype == torch.uint8:
                indices = indices.bool()
            if indices.dtype == torch.bool:
                if indices.shape == input_tensor.shape:
                    return getattr(torch._C._VariableFunctionsClass, "masked_select")(input_tensor, indices)
                else:
                    indices = getattr(torch._C._VariableFunctionsClass, "nonzero")(indices, as_tuple=True)
                    return getattr(torch._C._TensorBase, "__getitem__")(input_tensor, indices)
            elif indices.dtype != torch.bool:
                if not indices.shape or len(indices.shape) == 1:
                    return func(self, input_tensor, indices.tolist())
                elif len(indices.shape) == 2:
                    result = [func(self, input_tensor, index) for index in indices.tolist()]
                    return getattr(torch._C._VariableFunctionsClass, "stack")(result, 0)
                else:
                    res = [input_tensor[tensor_index] for tensor_index in indices]
                    return getattr(torch._C._VariableFunctionsClass, "stack")(res, 0)
        if self.op_name_ == "__eq__" and args[1] is None:
            return False
        return func(self, *args, **kwargs)
    return inner


def get_md5_for_tensor(x):
    if x.dtype == torch.bfloat16:
        x = x.float()
    tensor_bytes = x.cpu().detach().numpy().tobytes()
    crc32_hash = zlib.crc32(tensor_bytes)
    return f"{crc32_hash:08x}"
