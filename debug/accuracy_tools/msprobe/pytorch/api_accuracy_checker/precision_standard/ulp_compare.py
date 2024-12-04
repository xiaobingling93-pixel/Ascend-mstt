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

from collections import namedtuple
import numpy as np
import torch

from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import BaseConfig
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BasePrecisionCompare
from msprobe.core.common.const import CompareConst
from msprobe.pytorch.api_accuracy_checker.compare.algorithm import calc_ratio
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import ApiPrecisionCompareColumn, check_inf_or_nan, \
    is_inf_or_nan


UlpInfNanConsistency = namedtuple('UlpInfNanConsistency', ['ulp_inf_nan_consistency'])


class UlpPrecisionStandard(BasePrecisionCompare):
    def __init__(self, input_data):
        super().__init__(input_data)
        self.compare_algorithm = "ULP误差比对法"

    def _check_mean_ulp_err(self):
        column_name = ApiPrecisionCompareColumn.MEAN_ULP_ERR
        npu_value, gpu_value = self._get_and_convert_values(column_name)
        if is_inf_or_nan(npu_value) or is_inf_or_nan(gpu_value):
            return check_inf_or_nan(npu_value, gpu_value, column_name)
        else:
            return None, True, ""
        
    def _get_status(self, metrics, inf_nan_consistency):
        ulp_inf_nan_consistency = inf_nan_consistency.ulp_inf_nan_consistency
        _, mean_ulp_err_inf_nan_consistency, mean_ulp_err_message = self._check_mean_ulp_err()
        if not mean_ulp_err_inf_nan_consistency:
           metrics['compare_message'] = metrics.get('compare_message', '') + mean_ulp_err_message
        if not ulp_inf_nan_consistency or not mean_ulp_err_inf_nan_consistency:
            status_dict = {
                "ulp_err_status": CompareConst.ERROR
            }
            return CompareConst.ERROR, status_dict
        
        dtype = self.npu_precision.get(ApiPrecisionCompareColumn.DEVICE_DTYPE)
        if dtype == torch.float32:
            status, final_message = self._get_fp32_ulp_err_status()
        else:
            status, final_message = self._get_fp16_ulp_err_status()
        metrics['compare_message'] = metrics.get('compare_message', '') + final_message

        status_dict = {
            "ulp_err_status": status
        }
        return status, status_dict

    def _get_fp32_ulp_err_status(self):
        mean_ulp_err_threshold, ulp_err_proportion_threshold, ulp_err_proportion_ratio_threshold = \
                                                        BaseConfig.get_ulp_threshold(torch.float32)
        if self.mean_ulp_err < mean_ulp_err_threshold:
            return CompareConst.PASS, ""
        elif self.ulp_err_proportion < ulp_err_proportion_threshold:
            return CompareConst.PASS, ""
        elif self.ulp_err_proportion_ratio < ulp_err_proportion_ratio_threshold:
            return CompareConst.PASS, ""
        compare_message = "ERROR: ULP误差不满足标准\n"
        return CompareConst.ERROR, compare_message
        
    def _get_fp16_ulp_err_status(self):
        _, ulp_err_proportion_threshold, ulp_err_proportion_ratio_threshold = \
                                                        BaseConfig.get_ulp_threshold(torch.float16)
        if self.ulp_err_proportion < ulp_err_proportion_threshold:
            return CompareConst.PASS, ""
        elif self.ulp_err_proportion_ratio < ulp_err_proportion_ratio_threshold:
            return CompareConst.PASS, ""
        compare_message = "ERROR: ULP误差不满足标准\n"
        return CompareConst.ERROR, compare_message
    
    def _compute_ulp_err_proportion_ratio(self):
        column_name = ApiPrecisionCompareColumn.ULP_ERR_PROPORTION
        npu_value, gpu_value = self._get_and_convert_values(column_name)
        if is_inf_or_nan(npu_value) or is_inf_or_nan(gpu_value):
            return check_inf_or_nan(npu_value, gpu_value, column_name)
        else:
            return calc_ratio(npu_value, gpu_value, CompareConst.DEFAULT_RATIO_VALUE), True, ""
    
    def _compute_ratio(self):
        ulp_err_proportion_ratio, ulp_inf_nan_consistency, compare_message = self._compute_ulp_err_proportion_ratio()
        metrics = {
            "ulp_err_proportion_ratio": ulp_err_proportion_ratio,
            "compare_message": compare_message
            }
        return metrics, UlpInfNanConsistency(ulp_inf_nan_consistency)