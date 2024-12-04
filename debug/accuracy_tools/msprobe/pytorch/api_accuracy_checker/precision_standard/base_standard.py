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

from msprobe.pytorch.api_accuracy_checker.compare.algorithm import get_abs_bench_with_eps, get_abs_err, \
    get_finite_and_infinite_mask, get_small_value_mask
from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import StandardConfig


class BaseCompare:
    """
    Base comparison class for benchmarking and device output.

    This class provides a foundation for comparing benchmark outputs with device outputs.
    It encapsulates the common logic for calculating accuracy metrics and
    provides a framework for subclasses to implement specific comparison logic.

    Attributes:
        bench_output (np.ndarray): The output from the benchmark.
        device_output (np.ndarray): The output from the device.
        compare_column (object): The column object to store comparison results.
        dtype (torch.dtype): The data type of the outputs.

    Methods:
        get_small_value_threshold(): Retrieves the small value threshold for the given data type.
        stat_abs_bench_with_eps(): Calculates the absolute benchmark output with epsilon.
        stat_abs_error(): Calculates the absolute error between the benchmark and device outputs.
        stat_finite_and_infinite_mask(): Generates masks for finite and infinite/NaN values.
        stat_small_value_mask(abs_bench, both_finite_mask, small_value): Creates a mask for small values.
        compare(): Performs the comparison and computes metrics.
        _pre_compare(): Pre-comparison hook for subclass-specific initialization.
        _compute_metrics(): Computes the comparison metrics.
        _post_compare(metrics): Post-comparison hook to update comparison results.

    Note:
        This class assumes that the input data is an instance of InputData containing the benchmark output,
        device output, comparison column, and data type. Subclasses should implement the _pre_compare,
        _compute_metrics, and _post_compare methods to provide specific comparison logic.

    See Also:
        InputData: The class containing input data for comparison.
        StandardConfig: The class containing standard configuration values.
    """
    def __init__(self, input_data):
        self.bench_output = input_data.bench_output
        self.device_output = input_data.device_output

        self.compare_column = input_data.compare_column
        self.dtype = input_data.dtype

    @staticmethod
    def stat_small_value_mask(abs_bench, both_finite_mask, small_value):
        small_value_mask = get_small_value_mask(abs_bench, both_finite_mask, small_value)
        return small_value_mask

    def get_small_value_threshold(self):
        small_value = StandardConfig.get_small_valuel(self.dtype)
        small_value_atol = StandardConfig.get_small_value_atol(self.dtype)
        return small_value, small_value_atol
    
    def stat_abs_bench_with_eps(self):
        abs_bench, abs_bench_with_eps = get_abs_bench_with_eps(self.bench_output, self.dtype)
        return abs_bench, abs_bench_with_eps
    
    def stat_abs_error(self):
        abs_err = get_abs_err(self.bench_output, self.device_output)
        return abs_err
    
    def stat_finite_and_infinite_mask(self):
        both_finite_mask, inf_nan_mask = get_finite_and_infinite_mask(self.bench_output, self.device_output)
        return both_finite_mask, inf_nan_mask

    def compare(self):
        self._pre_compare()
        metrics = self._compute_metrics()
        self._post_compare(metrics)
    
    def _pre_compare(self):
        pass

    def _compute_metrics(self):
        return {}
    
    def _post_compare(self, metrics):
        self.compare_column.update(metrics)
