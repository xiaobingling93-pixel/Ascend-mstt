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

from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import StandardConfig
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BasePrecisionCompare
from msprobe.core.common.const import CompareConst
from msprobe.pytorch.api_accuracy_checker.compare.algorithm import calc_ratio
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import ApiPrecisionCompareColumn, check_inf_or_nan, \
    is_inf_or_nan


UlpInfNanConsistency = namedtuple('UlpInfNanConsistency', ['mean_ulp_err_inf_nan_consistency', 
                                             'ulp_err_proportion_ratio_inf_nan_consistency'])


class UlpPrecisionCompare(BasePrecisionCompare):
    def __init__(self, input_data):
        super().__init__(input_data)
        self.compare_algorithm = CompareConst.ULP_COMPARE_ALGORITHM_NAME

    @staticmethod
    def _compute_ulp_err_proportion_ratio(npu_value, gpu_value):
        column_name = ApiPrecisionCompareColumn.ULP_ERR_PROPORTION
        if is_inf_or_nan(npu_value) or is_inf_or_nan(gpu_value):
            return check_inf_or_nan(npu_value, gpu_value, column_name)
        else:
            return calc_ratio(npu_value, gpu_value, CompareConst.DEFAULT_RATIO_VALUE), True, ""

    def _compute_mean_ulp_err(self):
        column_name = ApiPrecisionCompareColumn.MEAN_ULP_ERR
        npu_value, gpu_value = self._get_and_convert_values(column_name)
        if is_inf_or_nan(npu_value) or is_inf_or_nan(gpu_value):
            _, mean_ulp_err_inf_nan_consistency, message = check_inf_or_nan(npu_value, gpu_value, column_name)
            return npu_value, mean_ulp_err_inf_nan_consistency, message
        else:
            return npu_value, True, ""
    
    def _compute_ulp_err_proportion(self):
        column_name = ApiPrecisionCompareColumn.ULP_ERR_PROPORTION
        npu_value, gpu_value = self._get_and_convert_values(column_name)
        return npu_value, gpu_value
        
    def _get_status(self, metrics, inf_nan_consistency):
        ulp_inf_nan_consistency = inf_nan_consistency.mean_ulp_err_inf_nan_consistency and \
                                  inf_nan_consistency.ulp_err_proportion_ratio_inf_nan_consistency  

        if not ulp_inf_nan_consistency:
            status_dict = {
                "ulp_err_status": CompareConst.ERROR
            }
            compare_result = CompareConst.ERROR
            metrics['compare_message'] =  metrics.get("compare_message", "") + "ERROR: ULP误差不满足标准\n"
            metrics.update({'compare_result': compare_result})
            return metrics
        
        dtype = self.row_npu.get(ApiPrecisionCompareColumn.DEVICE_DTYPE)
        mean_ulp_err = metrics.get("mean_ulp_error")
        ulp_err_proportion = metrics.get("ulp_err_proportion")
        ulp_err_proportion_ratio = metrics.get("ulp_err_proportion_ratio")
        if dtype == 'torch.float32':
            status, final_message = \
                self._get_fp32_ulp_err_status(mean_ulp_err, ulp_err_proportion, ulp_err_proportion_ratio)
        else:
            status, final_message = \
                self._get_fp16_ulp_err_status(ulp_err_proportion, ulp_err_proportion_ratio)
        metrics['compare_message'] =  metrics.get("compare_message", "") + final_message

        status_dict = {
            "ulp_err_status": status
        }
        compare_result = status
        metrics.update(status_dict)
        metrics.update({'compare_result': compare_result})
        return metrics

    def _get_fp32_ulp_err_status(self, mean_ulp_err, ulp_err_proportion, ulp_err_proportion_ratio):
        mean_ulp_err_threshold, ulp_err_proportion_threshold, ulp_err_proportion_ratio_threshold = \
                                                        StandardConfig.get_ulp_threshold(torch.float32)
        if mean_ulp_err < mean_ulp_err_threshold:
            return CompareConst.PASS, ""
        elif ulp_err_proportion < ulp_err_proportion_threshold:
            return CompareConst.PASS, ""
        elif ulp_err_proportion_ratio < ulp_err_proportion_ratio_threshold:
            return CompareConst.PASS, ""
        compare_message = "ERROR: ULP误差不满足标准\n"
        return CompareConst.ERROR, compare_message
        
    def _get_fp16_ulp_err_status(self, ulp_err_proportion, ulp_err_proportion_ratio):
        _, ulp_err_proportion_threshold, ulp_err_proportion_ratio_threshold = \
                                                        StandardConfig.get_ulp_threshold(torch.float16)
        if ulp_err_proportion < ulp_err_proportion_threshold:
            return CompareConst.PASS, ""
        elif ulp_err_proportion_ratio < ulp_err_proportion_ratio_threshold:
            return CompareConst.PASS, ""
        compare_message = "ERROR: ULP误差不满足标准\n"
        return CompareConst.ERROR, compare_message

    def _compute_ratio(self):
        compare_message = ""
        mean_ulp_err, mean_ulp_err_inf_nan_consistency, mean_ulp_err_message = self._compute_mean_ulp_err()
        compare_message += mean_ulp_err_message
        npu_ulp_err_proportion, gpu_ulp_err_proportion = self._compute_ulp_err_proportion()
        ulp_err_proportion_ratio, ulp_err_proportion_ratio_inf_nan_consistency, ulp_err_proportion_ratio_message = \
            self._compute_ulp_err_proportion_ratio(npu_ulp_err_proportion, gpu_ulp_err_proportion)
        compare_message += ulp_err_proportion_ratio_message
        metrics = {
            "mean_ulp_err": mean_ulp_err,
            "ulp_err_proportion": npu_ulp_err_proportion,
            "ulp_err_proportion_ratio": ulp_err_proportion_ratio,
            "compare_message": compare_message
        }
        return metrics, UlpInfNanConsistency(mean_ulp_err_inf_nan_consistency, 
                                             ulp_err_proportion_ratio_inf_nan_consistency)
