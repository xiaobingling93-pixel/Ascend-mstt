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


class BaseConfig:
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
    
    _small_value_threshold =  {
        'error_threshold': 2,
        'warning_threshold': 1,
        "default": 1
    }
    _rmse_threshold = {
        'error_threshold': 2,
        'warning_threshold': 1,
        "default": 1
    }
    _max_rel_err_threshold = {
        'error_threshold': 10,
        'warning_threshold': 1,
        "default": 1
    }
    _mean_rel_err_threshold = {
        'error_threshold': 2,
        'warning_threshold': 1,
        "default": 1
    }
    _eb_threshold ={
        'error_threshold': 2,
        'warning_threshold': 1,
        "default": 1
    }
    
    _fp32_mean_ulp_err_threshold = 64
    ulp_err_proportion_ratio = 1
    _fp32_ulp_err_proportion = 0.05
    _fp16_ulp_err_proportion = 0.001
    
    @classmethod
    def get_small_valuel(cls, dtype):
        return cls._small_value.get(dtype, cls._small_value["default"])
    
    @classmethod
    def get_small_value_atol(cls, dtype):
        return cls._small_value_atol.get(dtype, cls._small_value_atol["default"])
    
    @classmethod
    def get_rtol(cls, dtype):
        return cls._rtol.get(dtype, cls._rtol["default"])
    
    @classmethod
    def get_small_value_threshold(cls, threshold_type):
        return cls._small_value_threshold.get(threshold_type, cls._small_value_threshold["default"])
    
    @classmethod
    def get_rmse_threshold(cls, threshold_type):
        return cls._rmse_threshold.get(threshold_type, cls._rmse_threshold["default"])
    
    @classmethod
    def get_max_rel_err_threshold(cls, threshold_type):
        return cls._max_rel_err_threshold.get(threshold_type, cls._max_rel_err_threshold["default"])
    
    @classmethod
    def get_mean_rel_err_threshold(cls, threshold_type):
        return cls._mean_rel_err_threshold.get(threshold_type, cls._mean_rel_err_threshold["default"])
    
    @classmethod
    def get_eb_threshold(cls, threshold_type):
        return cls._eb_threshold.get(threshold_type, cls._eb_threshold["default"])
    
    def get_benchmark_threshold(metric):
        metric_threshold_functions = {
            'small_value': BaseConfig.get_small_value_threshold,
            'rmse': BaseConfig.get_rmse_threshold,
            'max_rel_err': BaseConfig.get_max_rel_err_threshold,
            'mean_rel_err': BaseConfig.get_mean_rel_err_threshold,
            'eb': BaseConfig.get_eb_threshold
        }
        
        threshold_func = metric_threshold_functions.get(metric)
        return threshold_func('error_threshold'), threshold_func('warning_threshold')

    @classmethod
    def get_fp32_mean_ulp_err_threshold(cls):
        return cls._fp32_mean_ulp_err_threshold

    @classmethod
    def get_ulp_err_proportion_ratio_threshold(cls):
        return cls.ulp_err_proportion_ratio

    @classmethod
    def get_fp32_ulp_err_proportion_threshold(cls):
        return cls._fp32_ulp_err_proportion

    @classmethod
    def get_fp16_ulp_err_proportion_threshold(cls):
        return cls._fp16_ulp_err_proportion
    
    def get_ulp_threshold(self, dtype):
        if dtype == torch.float32:
            mean_ulp_err_threshold = BaseConfig.get_fp32_mean_ulp_err_threshold()
            ulp_err_proportion_threshold = BaseConfig.get_fp32_ulp_err_proportion_threshold()
            ulp_err_proportion_ratio_threshold = BaseConfig.get_ulp_err_proportion_ratio_threshold()
            return mean_ulp_err_threshold, ulp_err_proportion_threshold, ulp_err_proportion_ratio_threshold
        else:
            ulp_err_proportion_threshold = BaseConfig.get_fp16_ulp_err_proportion_threshold()
            ulp_err_proportion_ratio_threshold = BaseConfig.get_ulp_err_proportion_ratio_threshold()
            return None, ulp_err_proportion_threshold, ulp_err_proportion_ratio_threshold
