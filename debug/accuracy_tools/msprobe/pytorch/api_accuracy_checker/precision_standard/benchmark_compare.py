import torch
import numpy as np

from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare
from msprobe.pytorch.api_accuracy_checker.compare.algorithm import get_small_value_err_ratio, get_rel_err, get_rmse, \
    get_error_balance, get_max_rel_err, get_mean_rel_err


class BenchmarkCompare(BaseCompare):

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
        small_value_err_ratio = get_small_value_err_ratio(self.small_value_mask, self.abs_err_greater_mask)
        rmse = get_rmse(self.abs_err, np.logical_or(self.inf_nan_mask, self.small_value_mask))
        eb = get_error_balance(self.bench_output, self.device_output)
        max_rel_error = get_max_rel_err(self.rel_err)
        mean_rel_error = get_mean_rel_err(self.rel_err)
        metrics = {
            "small_value_err_ratio": small_value_err_ratio,
            "max_rel_error": max_rel_error,
            "mean_rel_error": mean_rel_error,
            "rmse": rmse,
            "eb": eb
        }

        return metrics
