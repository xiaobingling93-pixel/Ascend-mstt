import numpy as np
from msprobe.pytorch.api_accuracy_checker.compare.algorithm import compare_bool_tensor
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare


class BinaryCompare(BaseCompare):
    """
    Binary comparison class for comparing boolean tensors.

    This class is designed to compare the output of a binary operation between a benchmark and a device.
    It calculates the error rate of the comparison and provides a simple metric for assessing the accuracy.

    Attributes:
        bench_output (np.ndarray): The output from the benchmark.
        device_output (np.ndarray): The output from the device.
        compare_column (object): The column object to store comparison results.
        dtype (torch.dtype): The data type of the outputs.

    Methods:
        _compute_metrics(): Computes the comparison metrics, specifically the error rate.

    Note:
        This class assumes that the input data is an instance of InputData containing the benchmark output,
        device output, comparison column, and data type. The outputs are expected to be boolean tensors.

    See Also:
        BaseCompare: The base class for comparison classes.
        compare_bool_tensor: The function used to compare boolean tensors.
    """
    def __init__(self, input_data):
        super(BinaryCompare, self).__init__(input_data)

    def _compute_metrics(self):
        error_rate, _, _ = compare_bool_tensor(self.bench_output, self.device_output)
        metrics = {
            "error_rate": error_rate
        }
        return metrics
