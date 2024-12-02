import numpy as np


class CompareInput:
    """
    A class to encapsulate the input data required for comparison operations.

    Attributes:
        bench_output (np.ndarray): The benchmark output values.
        device_output (np.ndarray): The device output values.
        compare_column (class): A clasee to store and update comparison metrics.
        dtype (type, optional): The data type of the outputs. Defaults to None.
        rel_err_orign (float or array-like, optional): The original relative error values. Defaults to None.

    Methods:
        __init__(bench_output, device_output, compare_column, dtype, rel_err_orign): Initializes an instance of CompareInput.
    """
    def __init__(self, bench_output, device_output, compare_column, dtype=None, rel_err_orign=None):
        self.bench_output = bench_output
        self.device_output = device_output
        if not isinstance(bench_output, np.ndarray) or not isinstance(device_output, np.ndarray):
            raise TypeError("The input should be numpy array")
        self.compare_column = compare_column
        self.dtype = dtype
        self.rel_err_orign = rel_err_orign
