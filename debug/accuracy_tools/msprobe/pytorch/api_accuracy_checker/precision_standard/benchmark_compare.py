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

from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare
from msprobe.pytorch.api_accuracy_checker.compare.algorithm import get_small_value_err_ratio, get_rel_err, get_rmse, \
    get_error_balance, get_max_rel_err, get_mean_rel_err


class BenchmarkCompare(BaseCompare):
    """
    Benchmark comparison class for calculating accuracy metrics.

    This class is designed to compare the output of a benchmark test with the output of a device.
    It calculates various metrics such as small value error ratio, RMSE, error balance, max relative error,
    and mean relative error to assess the accuracy of the device output against the benchmark output.

    Attributes:
        bench_output (np.ndarray): The output from the benchmark.
        device_output (np.ndarray): The output from the device.
        dtype (torch.dtype): The data type of the outputs.
        abs_bench (np.ndarray): The absolute value of the benchmark output.
        abs_bench_with_eps (np.ndarray): The absolute value of the benchmark output with epsilon.
        both_finite_mask (np.ndarray): A mask indicating where both outputs are finite.
        inf_nan_mask (np.ndarray): A mask indicating where either output is infinite or NaN.
        abs_err (np.ndarray): The absolute error between the benchmark and device outputs.
        small_value (float): The small value threshold for comparison.
        small_value_atol (float): The absolute tolerance for small values.
        small_value_mask (np.ndarray): A mask indicating where values are small.
        rel_err (np.ndarray): The relative error between the benchmark and device outputs.
        abs_err_greater_mask (np.ndarray): A mask indicating where absolute error is greater than the small value 
        tolerance.

    Methods:
        _get_abs_err_greater_mask(small_value_atol): Calculates a mask where absolute error is greater than the small 
        value tolerance.
        _compute_rel_err(): Computes the relative error between the benchmark and device outputs.
        _pre_compare(): Prepares the comparison by calculating various metrics.
        _compute_metrics(): Computes the accuracy metrics.

    Note:
        This class assumes that the input data is a dictionary containing 'bench_output', 'device_output', 
        'compare_column' and 'dtype'. 
        The data type should be a PyTorch data type.

    See Also:
        BaseCompare: The base class for comparison classes.
        InputData: The class containing input data for comparison.
    """

    def __init__(self, input_data):
        super(BenchmarkCompare, self).__init__(input_data)

    def _get_abs_err_greater_mask(self, small_value_atol):
        abs_err_greater_mask = np.greater(self.abs_err, small_value_atol)
        return abs_err_greater_mask
    
    def _compute_rel_err(self):
        rel_err = get_rel_err(self.abs_err, self.abs_bench_with_eps, self.small_value_mask, self.inf_nan_mask)
        return rel_err, rel_err.size
    
    def _pre_compare(self):
        self.abs_bench, self.abs_bench_with_eps = self.stat_abs_bench_with_eps()
        self.both_finite_mask, self.inf_nan_mask = self.stat_finite_and_infinite_mask()
        self.abs_err = self.stat_abs_error()
        self.small_value, self.small_value_atol = self.get_small_value_threshold()
        self.small_value_mask = self.stat_small_value_mask(self.abs_bench, self.both_finite_mask, self.small_value)
        self.rel_err, _ = self._compute_rel_err()
        self.abs_err_greater_mask = self._get_abs_err_greater_mask(self.small_value_atol)

    def _compute_metrics(self):
        """
        Computes a comprehensive set of error metrics for the comparison between benchmark and device outputs.

        This method calculates five key metrics:
        1. Small Value Error Ratio: The proportion of errors associated with small values.
        2. Root Mean Square Error (RMSE): The square root of the mean of the squared errors.
        3. Error Balance (EB): A measure of the balance between the errors in the benchmark and device outputs.
        4. Maximum Relative Error: The maximum relative error between the benchmark and device outputs.
        5. Mean Relative Error: The mean relative error between the benchmark and device outputs.
        
        Returns:
        dict: A dictionary containing the computed error metrics.
            The dictionary has the following keys:
            - "small_value_err_ratio": The proportion of errors associated with small values.
            - "max_rel_error": The maximum relative error.
            - "mean_rel_error": The mean relative error.
            - "rmse": The root mean square error.
            - "eb": The error balance.
        """
        small_value_err_ratio = get_small_value_err_ratio(self.small_value_mask, self.abs_err_greater_mask)
        rmse = get_rmse(self.abs_err, np.logical_or(self.inf_nan_mask, self.small_value_mask))
        eb = get_error_balance(self.bench_output, self.device_output)
        max_rel_error = get_max_rel_err(self.rel_err)
        mean_rel_error = get_mean_rel_err(self.rel_err)

        return {
            "small_value_err_ratio": small_value_err_ratio,
            "max_rel_error": max_rel_error,
            "mean_rel_error": mean_rel_error,
            "rmse": rmse,
            "eb": eb
        }
