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

from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import BaseConfig
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import convert_str_to_float
from msprobe.core.common.const import CompareConst


class BasePrecisionComare:
    def __init__(self, input_data):
        self.npu_precision = input_data.npu_precision
        self.gpu_precision = input_data.gpu_precision
        self.compare_column = input_data.compare_column
        self.compare_algorithm = None
    
    
    def _get_and_convert_values(self, column_name):
        npu_value = self.npu_precision.get(column_name)
        gpu_value = self.gpu_precision.get(column_name)
        npu_value = convert_str_to_float(npu_value)
        gpu_value = convert_str_to_float(gpu_value)
        return npu_value, gpu_value
    
    def get_status(self, metrics, inf_nan_consistency):
        pass

    def compute_ratio(self):
        pass
    
    def post_compare(self, metrics, inf_nan_consistency):
        compare_result, status_dict = self.get_status(metrics, inf_nan_consistency)
        metrics.update(status_dict)
        metrics.update({'compare_result': compare_result})
        metrics.update({'compare_algorithm': self.compare_algorithm})
        self.compare_column.update(metrics)
        return compare_result
    
    def compare(self):
        metrics, inf_nan_consistency = self.compute_ratio()
        compare_result = self.post_compare(metrics, inf_nan_consistency)
        return compare_result
