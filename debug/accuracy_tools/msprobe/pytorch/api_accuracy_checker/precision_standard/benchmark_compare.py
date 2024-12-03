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

import numpy as np
import math
from collections import namedtuple

from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import BaseConfig
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BasePrecisionComare
from msprobe.pytorch.api_accuracy_checker.compare.algorithm import calc_ratio
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import ApiPrecisionCompareColumn, check_inf_or_nan, \
    is_inf_or_nan
from msprobe.core.common.const import CompareConst


BenchmarkInfNanConsistency = namedtuple('BenchmarkInfNanConsistency', ['small_value_inf_nan_consistency', 
                                                                           'rmse_inf_nan_consistency', 
                                                                           'max_rel_inf_nan_consistency', 
                                                                           'mean_rel_inf_nan_consistency', 
                                                                           'eb_inf_nan_consistency'])


class BenchmarkPrecisionStandard(BasePrecisionComare):
    def __init__(self, input_data):
        super().__init__(input_data)
        self.compare_algorithm = "标杆比对法"

    @staticmethod
    def get_final_status(status_list):
        if CompareConst.ERROR in status_list:
            compare_result = CompareConst.ERROR
        elif CompareConst.WARNING in status_list:
            compare_result = CompareConst.WARNING
        return compare_result
    
    def _compute_small_value_err_ratio(self):
        column_name = ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE
        npu_value, gpu_value = self._get_and_convert_values(column_name)
        if is_inf_or_nan(npu_value) or is_inf_or_nan(gpu_value):
            return check_inf_or_nan(npu_value, gpu_value, column_name)
        else:
            return calc_ratio(npu_value, gpu_value, 10000.0), True, ""
    
    def _compute_rmse_ratio(self):
        column_name = ApiPrecisionCompareColumn.RMSE
        npu_value, gpu_value = self._get_and_convert_values(column_name)
        if is_inf_or_nan(npu_value) or is_inf_or_nan(gpu_value):
            return check_inf_or_nan(npu_value, gpu_value, column_name)
        else:
            return calc_ratio(npu_value, gpu_value, 10000.0), True, ""
    
    def _compute_max_rel_err_ratio(self):
        column_name = ApiPrecisionCompareColumn.MAX_REL_ERR
        npu_value, gpu_value = self._get_and_convert_values(column_name)
        if is_inf_or_nan(npu_value) or is_inf_or_nan(gpu_value):
            return check_inf_or_nan(npu_value, gpu_value, column_name)
        else:
            return calc_ratio(npu_value, gpu_value, 10000.0), True, ""
    
    def _compute_mean_rel_err_ratio(self):
        column_name = ApiPrecisionCompareColumn.MEAN_REL_ERR
        npu_value, gpu_value = self._get_and_convert_values(column_name)
        if is_inf_or_nan(npu_value) or is_inf_or_nan(gpu_value):
            return check_inf_or_nan(npu_value, gpu_value, column_name)
        else:
            return calc_ratio(npu_value, gpu_value, 10000.0), True, ""
    
    def _compute_eb_ratio(self):
        column_name = ApiPrecisionCompareColumn.EB
        npu_value, gpu_value = self._get_and_convert_values(column_name)
        if is_inf_or_nan(npu_value) or is_inf_or_nan(gpu_value):
            return check_inf_or_nan(npu_value, gpu_value, column_name)
        else:
            return calc_ratio(npu_value, gpu_value, 10000.0), True, ""
    
    def _compute_ratio(self):
        compare_message = ""
        small_value_err_ratio, small_value_inf_nan_consistency, small_value_message = self._compute_small_value_err_ratio()
        compare_message += small_value_message
        rmse_ratio, rmse_inf_nan_consistency, rmse_message = self._compute_rmse_ratio()
        compare_message += rmse_message
        max_rel_err_ratio, max_rel_inf_nan_consistency, max_rel_message = self._compute_max_rel_err_ratio()
        compare_message += max_rel_message
        mean_rel_err_ratio, mean_rel_inf_nan_consistency, mean_rel_message = self._compute_mean_rel_err_ratio()
        compare_message += mean_rel_message
        eb_ratio, eb_inf_nan_consistency, eb_message = self._compute_eb_ratio()
        compare_message += eb_message
        
        metrics = {
            "small_value_err_ratio": small_value_err_ratio,
            "rmse_ratio": rmse_ratio,
            "max_rel_err_ratio": max_rel_err_ratio,
            "mean_rel_err_ratio": mean_rel_err_ratio,
            "eb_ratio": eb_ratio,
            "compare_message": compare_message
        }
        
        return metrics, \
               BenchmarkInfNanConsistency(small_value_inf_nan_consistency, rmse_inf_nan_consistency, 
                                          max_rel_inf_nan_consistency, mean_rel_inf_nan_consistency,
                                          eb_inf_nan_consistency)
    
    
    def _get_threshold(self, metric):
        error_threshold, warning_threshold = BaseConfig.get_threshold(metric)
        return error_threshold, warning_threshold
    

    def _get_single_metric_status(self, ratio, metric):
        if math.isnan(ratio) or math.isinf(ratio):
            return CompareConst.PASS
        error_threshold, warning_threshold = self._get_threshold(metric)
        if ratio > error_threshold:
            return CompareConst.ERROR
        elif ratio > warning_threshold:
            return CompareConst.WARNING
        return CompareConst.PASS

    def _get_status(self, metrics, inf_nan_consistency):
        small_value_inf_nan_consistency = inf_nan_consistency.small_value_inf_nan_consistency
        rmse_inf_nan_consistency = inf_nan_consistency.rmse_inf_nan_consistency
        max_rel_inf_nan_consistency = inf_nan_consistency.max_rel_inf_nan_consistency
        mean_rel_inf_nan_consistency = inf_nan_consistency.mean_rel_inf_nan_consistency
        eb_inf_nan_consistency = inf_nan_consistency.eb_inf_nan_consistency
        small_value_err_ratio = metrics.get("small_value_err_ratio")
        rmse_ratio = metrics.get("rmse_ratio")  
        max_rel_err_ratio = metrics.get("max_rel_err_ratio")
        mean_rel_err_ratio = metrics.get("mean_rel_err_ratio")
        eb_ratio = metrics.get("eb_ratio")
        
        small_value_err_status = self._get_single_metric_status(small_value_err_ratio, 'small_value') \
                                        if small_value_inf_nan_consistency else CompareConst.ERROR
        rmse_status = self._get_single_metric_status(rmse_ratio, 'rmse')  \
                                        if rmse_inf_nan_consistency else CompareConst.ERROR
        max_rel_err_status = self._get_single_metric_status(max_rel_err_ratio,'max_rel_err') \
                                        if max_rel_inf_nan_consistency else CompareConst.ERROR
        mean_rel_err_status = self._get_single_metric_status(mean_rel_err_ratio,'mean_rel_err') \
                                        if mean_rel_inf_nan_consistency else CompareConst.ERROR
        eb_status = self._get_single_metric_status(eb_ratio, 'eb') \
                                        if eb_inf_nan_consistency else CompareConst.ERROR
        status_list = [small_value_err_status, rmse_status, max_rel_err_status, mean_rel_err_status, eb_status]
        compare_result = self.get_final_status(status_list)
        status_dict = {
            "small_value_err_status": small_value_err_status,
            "rmse_status": rmse_status,
            "max_rel_err_status": max_rel_err_status,
            "mean_rel_err_status": mean_rel_err_status,
            "eb_status": eb_status
        }
        return compare_result, status_dict