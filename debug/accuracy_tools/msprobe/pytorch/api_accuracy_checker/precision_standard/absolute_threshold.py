import torch
import numpy as np

from msprobe.pytorch.api_accuracy_checker.compare.algorithm import check_inf_nan_value, check_norm_value, \
    check_small_value
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare
from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import StandardConfig



class AbsolutethdCompare(BaseCompare):
    def __init__(self, input_data):
        super(AbsolutethdCompare, self).__init__(input_data)

    def _get_rtol(self):
        return StandardConfig.get_rtol(self.dtype)

    def _get_rel_err(self, abs_bench_with_eps):
        abs_err = self.stat_abs_error()
        rel_err = abs_err / abs_bench_with_eps
        return rel_err

    def _get_normal_value_mask(self, small_value_mask):
        return np.logical_and(self.both_finite_mask, np.logical_not(small_value_mask))

    def _pre_compare(self):
        self.abs_bench, self.abs_bench_with_eps = self.stat_abs_bench_with_eps()
        self.both_finite_mask, self.inf_nan_mask = self.stat_finite_and_infinite_mask()
        self.rtol = self._get_rtol()
        self.rel_err = self._get_rel_err(self.abs_bench_with_eps)
        self.small_value, self.small_value_atol = self.get_small_value_threshold()
        self.small_value_mask = self.stat_small_value_mask(self.abs_bench, self.both_finite_mask, self.small_value)
        self.normal_value_mask = self._get_normal_value_mask(self.small_value_mask)

    def _compute_metrics(self):
        inf_nan_error_ratio = check_inf_nan_value(self.inf_nan_mask, self.bench_output, self.device_output, self.dtype,
                                                  self.rtol)
        rel_err_ratio = check_norm_value(self.normal_value_mask, self.rel_err, self.rtol)
        abs_err_ratio = check_small_value(self.abs_bench, self.both_finite_mask, self.small_value_atol)
        metrics = {
            "inf_nan_error_ratio": inf_nan_error_ratio,
            "rel_err_ratio": rel_err_ratio,
            "abs_err_ratio": abs_err_ratio
        }
        return metrics
