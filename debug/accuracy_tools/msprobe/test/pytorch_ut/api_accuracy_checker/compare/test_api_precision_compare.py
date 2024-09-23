import unittest

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
        
        self.api_name = "test_api"
        self.npu_precision = {'ERROR_RATE': '0', 'SMALL_VALUE_ERROR_RATE': '0.01', 'RMSE': '0.1', 'MAX_REL_ERR': '0.1',
                              'MEAN_REL_ERR': '0.1', 'EB': '0.1', 'MEAN_ULP_ERR': '0.1', 'ULP_ERR_PROPORTION': '0.05'}
        self.gpu_precision = {'ERROR_RATE': '0', 'SMALL_VALUE_ERROR_RATE': '0.01', 'RMSE': '0.1', 'MAX_REL_ERR': '0.1',
                              'MEAN_REL_ERR': '0.1', 'EB': '0.1', 'MEAN_ULP_ERR': '0.2', 'ULP_ERR_PROPORTION': '0.06'}
        self.ulp_standard = ULPStandard(self.api_name, self.npu_precision, self.gpu_precision)
        self.benchmark_standard = BenchmarkStandard(self.api_name, self.npu_precision, self.gpu_precision)

    def test_benchmark_standard_calc_ratio(self):
        column_name = "TEST_COLUMN"
        default_value = 0
        result = BenchmarkStandard._calc_ratio(column_name, '2', '1', default_value)
        self.assertEqual(result[0], 2.0)

        result = BenchmarkStandard._calc_ratio(column_name, '0', '0', default_value)
        self.assertEqual(result[0], 1.0)

        result = BenchmarkStandard._calc_ratio(column_name, '1', '0', default_value)
        self.assertEqual(result[0], default_value)

        result = BenchmarkStandard._calc_ratio(column_name, 'nan', '0', default_value)
        self.assertTrue(math.isnan(result[0]))

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

    def test_write_detail_csv(self):
        content = [1, 2, 3]
        save_path = "path/temp.csv"
        write_detail_csv(content, save_path)
        self.assertTrue(os.path.exists(save_path))
        # os.rmdir(save_path)
        
    def test_ulp_standard(self):
        # 测试 ULPStandard 函数
        self.ulp_standard.get_result()
        self.assertEqual(self.ulp_standard.ulp_err_status, CompareConst.PASS)

    def test_benchmark_standard(self):
        # 测试 BenchmarkStandard 函数
        self.benchmark_standard.get_result()
        self.assertEqual(self.benchmark_standard.final_result, CompareConst.PASS)


if __name__ == '__main__':
    unittest.main()
