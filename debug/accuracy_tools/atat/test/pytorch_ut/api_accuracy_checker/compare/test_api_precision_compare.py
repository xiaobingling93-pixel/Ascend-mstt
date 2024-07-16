import unittest

import pandas as pd

from atat.pytorch.api_accuracy_checker.compare.api_precision_compare import (
    CompareConfig,
    BenchmarkStandard,
    check_csv_columns,
    check_error_rate,
    get_api_checker_result,
)
from atat.pytorch.api_accuracy_checker.compare.compare_utils import CompareConst


class TestApiPrecisionCompare(unittest.TestCase):

    def setUp(self):
        # Setup paths and mock data
        self.config = CompareConfig(
            npu_csv_path='mock_npu.csv',
            gpu_csv_path='mock_gpu.csv',
            result_csv_path='result.csv',
            details_csv_path='details.csv'
        )

        self.npu_data = pd.DataFrame({
            'API_NAME': ['api1.forward', 'api1.backward'],
            'DEVICE_DTYPE': ['float32', 'float32'],
            'ERROR_RATE': ['0', '0.1'],
            'SMALL_VALUE_ERROR_RATE': ['0.01', '0.02'],
            'RMSE': ['0.1', '0.2'],
            'MAX_REL_ERR': ['0.1', '0.2'],
            'MEAN_REL_ERR': ['0.1', '0.2'],
            'EB': ['0.1', '0.2']
        })

        self.gpu_data = pd.DataFrame({
            'API_NAME': ['api1.forward', 'api1.backward'],
            'DEVICE_DTYPE': ['float32', 'float32'],
            'ERROR_RATE': ['0', '0'],
            'SMALL_VALUE_ERROR_RATE': ['0.01', '0.01'],
            'RMSE': ['0.1', '0.1'],
            'MAX_REL_ERR': ['0.1', '0.1'],
            'MEAN_REL_ERR': ['0.1', '0.1'],
            'EB': ['0.1', '0.1']
        })

    def test_benchmark_standard_calc_ratio(self):
        column_name = "TEST_COLUMN"
        default_value = 0
        result = BenchmarkStandard._calc_ratio(column_name, '2', '1', default_value)
        self.assertEqual(result[0], 2.0)

        result = BenchmarkStandard._calc_ratio(column_name, '0', '0', default_value)
        self.assertEqual(result[0], 1.0)

    def test_check_csv_columns(self):
        with self.assertRaises(Exception):
            check_csv_columns([], 'test_csv')

    def test_check_error_rate(self):
        result = check_error_rate('0')
        self.assertEqual(result, CompareConst.PASS)

        result = check_error_rate('0.1')
        self.assertEqual(result, CompareConst.ERROR)

    def test_get_api_checker_result(self):
        result = get_api_checker_result([CompareConst.PASS, CompareConst.ERROR])
        self.assertEqual(result, CompareConst.ERROR)

        result = get_api_checker_result([CompareConst.PASS, CompareConst.PASS])
        self.assertEqual(result, CompareConst.PASS)


if __name__ == '__main__':
    unittest.main()
