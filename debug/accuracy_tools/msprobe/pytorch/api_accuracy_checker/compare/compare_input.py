import numpy as np


class CompareInput:
    def __init__(self, bench_output, device_output, compare_column, dtype=None, rel_err_orign=None):
        self.bench_output = bench_output
        self.device_output = device_output
        if not isinstance(bench_output, np.ndarray) or not isinstance(device_output, np.ndarray):
            raise TypeError("The input should be numpy array")
        self.compare_column = compare_column
        self.dtype = dtype
        self.rel_err_orign = rel_err_orign
