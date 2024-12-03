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

import torch


class StandardConfig:
    """
    Standard configuration class for managing precision and comparison thresholds.

    This class provides a centralized way to manage the small value thresholds, absolute tolerances,
    and relative tolerances (rtol) used in precision comparisons. It allows for different thresholds
    based on the data type, with default values provided for common data types.

    Attributes:
        _small_value (dict): A dictionary mapping data types to their corresponding small value thresholds.
        _small_value_atol (dict): A dictionary mapping data types to their corresponding absolute tolerances.
        _rtol (dict): A dictionary mapping data types to their corresponding relative tolerances.

    Methods:
        get_small_value(dtype): Retrieves the small value threshold for the given data type.
        get_small_value_atol(dtype): Retrieves the absolute tolerance for the given data type.
        get_rtol(dtype): Retrieves the relative tolerance for the given data type.

    Example:
        >>> small_value = StandardConfig.get_small_value(torch.float32)
        >>> atol = StandardConfig.get_small_value_atol(torch.float32)
        >>> rtol = StandardConfig.get_rtol(torch.float32)
        >>> print(small_value, atol, rtol)
        1e-6 1e-9 1e-6

    Note:
        The data type is expected to be a PyTorch data type. If the data type is not found in the dictionary,
        the default value is returned.

    See Also:
        torch.dtype: PyTorch data types.
    """
    _small_value = {
        torch.float16: 1e-3,
        torch.bfloat16: 1e-3,
        torch.float32: 1e-6,
        "default": 1e-6
        }
    _small_value_atol = {
        torch.float16: 1e-5,
        torch.bfloat16: 1e-5,
        torch.float32: 1e-9,
        "default": 1e-9
        }
    _rtol = {
        torch.float16: 1e-3,
        torch.bfloat16: 4e-3,
        torch.float32: 1e-6,
        "default": 1e-6  # 默认值也放在配置类中
    }

    @classmethod
    def get_small_valuel(cls, dtype):
        return cls._small_value.get(dtype, cls._small_value["default"])
    
    @classmethod
    def get_small_value_atol(cls, dtype):
        return cls._small_value_atol.get(dtype, cls._small_value_atol["default"])
    
    @classmethod
    def get_rtol(cls, dtype):
        return cls._rtol.get(dtype, cls._rtol["default"])
