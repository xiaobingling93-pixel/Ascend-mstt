from msprobe.pytorch.api_accuracy_checker.compare.algorithm import get_abs_bench_with_eps, get_abs_err, \
    get_finite_and_infinite_mask, get_small_value_mask
from msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config import StandardConfig


class BaseCompare:
    def __init__(self, input_data):
        self.bench_output = input_data.bench_output
        self.device_output = input_data.device_output

        self.compare_column = input_data.compare_column
        self.dtype = input_data.dtype

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
    
    def stat_small_value_mask(self, abs_bench, both_finite_mask, small_value):
        small_value_mask = get_small_value_mask(abs_bench, both_finite_mask, small_value)
        return small_value_mask

    def compare(self):
        self._pre_compare()
        metrics = self._compute_metrics()
        self._post_compare(metrics)
    
    def _pre_compare(self):
        pass

    def _compute_metrics(self):
        metrics = {}
        return metrics
    
    def _post_compare(self, metrics):
        self.compare_column.update(metrics)
