import numpy as np
from msprobe.pytorch.api_accuracy_checker.compare.algorithm import compare_bool_tensor
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare


class BinaryCompare(BaseCompare):

    def __init__(self, input_data):
        super(BinaryCompare, self).__init__(input_data)

    def _compute_metrics(self):
        error_rate, _, _ = compare_bool_tensor(self.bench_output, self.device_output)
        metrics = {
            "error_rate": error_rate
        }
        return metrics
