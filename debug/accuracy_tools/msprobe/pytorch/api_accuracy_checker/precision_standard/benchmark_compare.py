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

import math
from collections import namedtuple
import numpy as np

from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import StandardConfig
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BasePrecisionCompare
from msprobe.pytorch.api_accuracy_checker.compare.algorithm import calc_ratio
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import ApiPrecisionCompareColumn, check_inf_or_nan, \
    is_inf_or_nan
from msprobe.core.common.const import CompareConst


BenchmarkInfNanConsistency = namedtuple('BenchmarkInfNanConsistency', ['small_value_inf_nan_consistency', 
                                                                           'rmse_inf_nan_consistency', 
                                                                           'max_rel_inf_nan_consistency', 
                                                                           'mean_rel_inf_nan_consistency', 
                                                                           'eb_inf_nan_consistency'])


class BenchmarkPrecisionCompare(BasePrecisionCompare):
    def __init__(self, input_data):
        super().__init__(input_data)
        self.compare_algorithm = CompareConst.BENCHMARK_COMPARE_ALGORITHM_NAME

    @staticmethod
    def get_final_status(status_list):
        compare_result = CompareConst.PASS
        if CompareConst.ERROR in status_list:
            compare_result = CompareConst.ERROR
        elif CompareConst.WARNING in status_list:
            compare_result = CompareConst.WARNING
        return compare_result
    
    def _calc_ratio(self, column_name):
        npu_value, gpu_value = self._get_and_convert_values(column_name)
        if is_inf_or_nan(npu_value) or is_inf_or_nan(gpu_value):
            return check_inf_or_nan(npu_value, gpu_value, column_name)
        else:
            return calc_ratio(npu_value, gpu_value, CompareConst.DEFAULT_RATIO_VALUE), True, ""

    def _compute_ratio(self):
        compare_message = ""
        small_value_err_ratio, small_value_inf_nan_consistency, small_value_message = \
            self._calc_ratio(ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE)
        compare_message += small_value_message
        rmse_ratio, rmse_inf_nan_consistency, rmse_message = self._calc_ratio(ApiPrecisionCompareColumn.RMSE)
        compare_message += rmse_message
        max_rel_err_ratio, max_rel_inf_nan_consistency, max_rel_message = \
            self._calc_ratio(ApiPrecisionCompareColumn.MAX_REL_ERR)
        compare_message += max_rel_message
        mean_rel_err_ratio, mean_rel_inf_nan_consistency, mean_rel_message = \
            self._calc_ratio(ApiPrecisionCompareColumn.MEAN_REL_ERR)
        compare_message += mean_rel_message
        eb_ratio, eb_inf_nan_consistency, eb_message = self._calc_ratio(ApiPrecisionCompareColumn.EB)
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
        error_threshold, warning_threshold = StandardConfig.get_benchmark_threshold(metric)
        return error_threshold, warning_threshold

    def _get_single_metric_status(self, ratio, metric):
        if is_inf_or_nan(ratio):
            return CompareConst.PASS
        error_threshold, warning_threshold = self._get_threshold(metric)
        if ratio > error_threshold:
            return CompareConst.ERROR
        elif ratio > warning_threshold:
            return CompareConst.WARNING
        return CompareConst.PASS

    def _get_status(self, metrics, inf_nan_consistency):
        small_value_err_ratio = metrics.get(CompareConst.SMALL_VALUE_ERR_RATIO)
        rmse_ratio = metrics.get(CompareConst.RMSE_RATIO)
        max_rel_err_ratio = metrics.get(CompareConst.MAX_REL_ERR_RATIO)
        mean_rel_err_ratio = metrics.get(CompareConst.MEAN_REL_ERR_RATIO)
        eb_ratio = metrics.get(CompareConst.EB_RATIO)
        
        small_value_err_status = self._get_single_metric_status(small_value_err_ratio, CompareConst.SMALL_VALUE) \
                                        if inf_nan_consistency.small_value_inf_nan_consistency else CompareConst.ERROR
        rmse_status = self._get_single_metric_status(rmse_ratio, CompareConst.RMSE)  \
                                        if inf_nan_consistency.rmse_inf_nan_consistency else CompareConst.ERROR
        max_rel_err_status = self._get_single_metric_status(max_rel_err_ratio, CompareConst.MAX_REL_ERR) \
                                        if inf_nan_consistency.max_rel_inf_nan_consistency else CompareConst.ERROR
        mean_rel_err_status = self._get_single_metric_status(mean_rel_err_ratio, CompareConst.MEAN_REL_ERR) \
                                        if inf_nan_consistency.mean_rel_inf_nan_consistency else CompareConst.ERROR
        eb_status = self._get_single_metric_status(eb_ratio, CompareConst.EB) \
                                        if inf_nan_consistency.eb_inf_nan_consistency else CompareConst.ERROR
        status_list = [small_value_err_status, rmse_status, max_rel_err_status, mean_rel_err_status, eb_status]
        compare_result = self.get_final_status(status_list)
        status_dict = {
            CompareConst.SMALL_VALUE_ERR_STATUS: small_value_err_status,
            CompareConst.RMSE_STATUS: rmse_status,
            CompareConst.MAX_REL_ERR_STATUS: max_rel_err_status,
            CompareConst.MEAN_REL_ERR_STATUS: mean_rel_err_status,
            CompareConst.EB_STATUS: eb_status
        }
        metrics.update(status_dict)
        metrics.update({CompareConst.COMPARE_RESULT: compare_result})
        return metrics
