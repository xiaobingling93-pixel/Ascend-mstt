import numpy as np
import torch

from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare
from msprobe.pytorch.api_accuracy_checker.compare.algorithm import get_ulp_err
from msprobe.core.common.const import CompareConst


class UlpCompare(BaseCompare):
    """
    Ulp compare comparison class for calculating accuracy metrics.

    Attributes:
        bench_output (array-like): The benchmark output values.
        device_output (array-like): The device output values.
        dtype (torch.dtype): The data type of the outputs (e.g., torch.float32 or torch.float16).
        ulp_err (array-like): The ULP errors calculated from the benchmark and device outputs.

    Methods:
        _stat_max_ulp_err(ulp_err): Calculates the maximum ULP error.
        _stat_mean_ulp_err(ulp_err): Calculates the mean ULP error.
        _stat_ulp_error_proportion(ulp_err): Calculates the proportion of ULP errors exceeding a threshold.
        _pre_compare(): Prepares for comparison by calculating ULP errors.
        _compute_metrics(): Computes the ULP error metrics.
    """
    def __init__(self, input_data):
        super(UlpCompare, self).__init__(input_data)
    
    @staticmethod
    def _stat_max_ulp_err(ulp_err):
        return np.max(ulp_err)
    
    @staticmethod
    def _stat_mean_ulp_err(ulp_err):
        return np.mean(ulp_err)
    
    def _stat_ulp_error_proportion(self, ulp_err):
        if self.dtype == torch.float32:
            return np.sum(ulp_err > CompareConst.ULP_FLOAT32_THRESHOLD) / self.bench_output.size
        else:
            return np.sum(ulp_err > CompareConst.ULP_FLOAT16_THRESHOLD) / self.bench_output.size
    
    def _pre_compare(self):
        self.ulp_err = get_ulp_err(self.bench_output, self.device_output, self.dtype)
    
    def _compute_metrics(self):
        max_ulp_error = self._stat_max_ulp_err(self.ulp_err)
        mean_ulp_error = self._stat_mean_ulp_err(self.ulp_err)
        
        ulp_error_proportion = self._stat_ulp_error_proportion(self.ulp_err)
        
        metrics = {
            "max_ulp_error": max_ulp_error,
            "mean_ulp_error": mean_ulp_error,
            "ulp_error_proportion": ulp_error_proportion
        }
        
        return metrics
