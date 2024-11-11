import unittest
from unittest.mock import patch

import pandas as pd

from msprobe.pytorch.api_accuracy_checker.compare.api_precision_compare import *
from msprobe.core.common.exceptions import FileCheckException
from msprobe.pytorch.api_accuracy_checker.compare.api_precision_compare import _api_precision_compare_command
from msprobe.core.common.const import CompareConst


class Args:
    def __init__(self, npu_csv_path=None, gpu_csv_path=None, out_path=None):
        self.npu_csv_path = npu_csv_path
        self.gpu_csv_path = gpu_csv_path
        self.out_path = out_path


class TestFileCheck(unittest.TestCase):
    def setUp(self):
        src_path = 'temp_path'
        create_directory(src_path)
        dst_path = 'soft_link'
        os.symlink(src_path, dst_path)
        self.hard_path = os.path.abspath(src_path)
        self.soft_path = os.path.abspath(dst_path)
        csv_path = os.path.join(self.hard_path, 'test.csv')
        csv_data = [['1', '2', '3'], ['4', '5', '6']]
        write_csv(csv_path, csv_data)
        self.hard_csv_path = os.path.abspath(csv_path)
        soft_csv_path = os.path.join(self.hard_path, 'soft.csv')
        os.symlink(csv_path, soft_csv_path)
        self.soft_csv_path = os.path.abspath(soft_csv_path)

    def tearDown(self):
        for file in os.listdir(self.hard_path):
            os.remove(os.path.join(self.hard_path, file))
        os.rmdir(self.soft_path)
        os.rmdir(self.hard_path)
        os.rmdir(self.hard_csv_path)
        os.rmdir(self.soft_csv_path)

    def test_npu_path_check(self):
        args = Args(npu_csv_path=self.soft_csv_path, gpu_csv_path=self.hard_csv_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            _api_precision_compare_command(args)
        self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_gpu_path_check(self):
        args = Args(npu_csv_path=self.hard_csv_path, gpu_csv_path=self.soft_csv_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            _api_precision_compare_command(args)
        self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_out_path_check(self):
        args = Args(npu_csv_path=self.hard_csv_path, gpu_csv_path=self.hard_csv_path, out_path=self.soft_path)
        
        with self.assertRaises(Exception) as context:
            _api_precision_compare_command(args)
        self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)


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
        self.npu_precision = {
            ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO: '0', ApiPrecisionCompareColumn.REL_ERR_RATIO: '0',
            ApiPrecisionCompareColumn.ABS_ERR_RATIO: '0', ApiPrecisionCompareColumn.ERROR_RATE: '0', 
            ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE: '0.01', ApiPrecisionCompareColumn.RMSE: '0.1', 
            ApiPrecisionCompareColumn.MAX_REL_ERR: '0.1', ApiPrecisionCompareColumn.MEAN_REL_ERR: '0.1', 
            ApiPrecisionCompareColumn.EB: '0.1', ApiPrecisionCompareColumn.MEAN_ULP_ERR: '0.1', 
            ApiPrecisionCompareColumn.ULP_ERR_PROPORTION: '0.05'
            }
        self.gpu_precision = {
            ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO: '0', ApiPrecisionCompareColumn.REL_ERR_RATIO: '0',
            ApiPrecisionCompareColumn.ABS_ERR_RATIO: '0', ApiPrecisionCompareColumn.ERROR_RATE: '0', 
            ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE: '0.01', ApiPrecisionCompareColumn.RMSE: '0.1', 
            ApiPrecisionCompareColumn.MAX_REL_ERR: '0.1', ApiPrecisionCompareColumn.MEAN_REL_ERR: '0.1', 
            ApiPrecisionCompareColumn.EB: '0.1', ApiPrecisionCompareColumn.MEAN_ULP_ERR: '0.2', 
            ApiPrecisionCompareColumn.ULP_ERR_PROPORTION: '0.06'}
        
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

        result = get_api_checker_result([])
        self.assertEqual(result, CompareConst.SPACE)

        result = get_api_checker_result([CompareConst.SKIP])
        self.assertEqual(result, CompareConst.SKIP)

    def test_write_detail_csv(self):
        content = [1, 2, 3]
        save_path = "path/"
        create_directory(save_path)
        details_csv_path = os.path.join(save_path, "details.csv")
        write_detail_csv(content, details_csv_path)
        self.assertTrue(os.path.exists(details_csv_path))
        for filename in os.listdir(save_path):
            os.remove(os.path.join(save_path, filename))
        os.rmdir(save_path)
        
    def test_ulp_standard(self):
        self.ulp_standard.get_result()
        self.assertEqual(self.ulp_standard.ulp_err_status, CompareConst.PASS)

        self.assertEqual(self.ulp_standard._get_ulp_status(torch.float32), CompareConst.PASS)

    def test_benchmark_standard(self):
        self.benchmark_standard.get_result()
        self.assertEqual(self.benchmark_standard.final_result, CompareConst.PASS)

        column_list = self.benchmark_standard.to_column_value()
        expect_column_list = [1, 'pass', 1, 'pass', 1, 'pass', 1, 'pass', 1, 'pass']
        self.assertEqual(column_list, expect_column_list)
        
    def test_get_absolute_threshold_result_pass(self):
        row_npu = {
            ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO: '0',
            ApiPrecisionCompareColumn.REL_ERR_RATIO: '0',
            ApiPrecisionCompareColumn.ABS_ERR_RATIO: '0'
        }
        result = get_absolute_threshold_result(row_npu)
        self.assertEqual(result['inf_nan_error_ratio'], 0.0)
        self.assertEqual(result['inf_nan_result'], CompareConst.PASS)
        self.assertEqual(result['rel_err_ratio'], 0.0)
        self.assertEqual(result['rel_err_result'], CompareConst.PASS)
        self.assertEqual(result['abs_err_ratio'], 0.0)
        self.assertEqual(result['abs_err_result'], CompareConst.PASS)
        self.assertEqual(result['absolute_threshold_result'], CompareConst.PASS)

    def test_get_absolute_threshold_result_error(self):
        row_npu = {
            ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO: '0',
            ApiPrecisionCompareColumn.REL_ERR_RATIO: '0.1',
            ApiPrecisionCompareColumn.ABS_ERR_RATIO: '0'
        }
        result = get_absolute_threshold_result(row_npu)
        self.assertEqual(result['inf_nan_error_ratio'], 0.0)
        self.assertEqual(result['inf_nan_result'], CompareConst.PASS)
        self.assertEqual(result['rel_err_ratio'], 0.1)
        self.assertEqual(result['rel_err_result'], CompareConst.ERROR)
        self.assertEqual(result['abs_err_ratio'], 0.0)
        self.assertEqual(result['abs_err_result'], CompareConst.PASS)
        self.assertEqual(result['absolute_threshold_result'], CompareConst.ERROR)


if __name__ == '__main__':
    unittest.main()
