import unittest
from unittest.mock import patch

import pandas as pd

from msprobe.pytorch.api_accuracy_checker.compare.api_precision_compare import *
from msprobe.core.common.const import CompareConst


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
    
    def test_print_test_success_success(self):
        with patch('msprobe.pytorch.common.log.logger.info') as mock_info:
            api_full_name = "test_api"
            forward_result = CompareConst.PASS
            backward_result = CompareConst.PASS
            print_test_success(api_full_name, forward_result, backward_result)
            mock_info.assert_called_once_with(f"running api_full_name {api_full_name} compare, "
                                              f"is_fwd_success: True, "
                                              f"is_bwd_success: True")

    def test_print_test_success_forward_failure(self):
        with patch('msprobe.pytorch.common.log.logger.info') as mock_info:
            api_full_name = "test_api"
            forward_result = CompareConst.ERROR
            backward_result = CompareConst.PASS
            print_test_success(api_full_name, forward_result, backward_result)
            mock_info.assert_called_once_with(f"running api_full_name {api_full_name} compare, "
                                              f"is_fwd_success: False, "
                                              f"is_bwd_success: True")

    def test_print_test_success_backward_failure(self):
        with patch('msprobe.pytorch.common.log.logger.info') as mock_info:
            api_full_name = "test_api"
            forward_result = CompareConst.PASS
            backward_result = CompareConst.ERROR
            print_test_success(api_full_name, forward_result, backward_result)
            mock_info.assert_called_once_with(f"running api_full_name {api_full_name} compare, "
                                              f"is_fwd_success: True, "
                                              f"is_bwd_success: False")


if __name__ == '__main__':
    unittest.main()
