import unittest
import numpy as np
from msprobe.pytorch.api_accuracy_checker.precision_standard.benchmark_compare import BenchmarkPrecisionCompare
from msprobe.core.common.const import CompareConst

class TestBenchmarkPrecisionCompare(unittest.TestCase):
    def setUp(self):
        # 准备真实的输入数据
        self.input_data = {
            'npu': {
                'small_value_error_rate': 0.001,
                'rmse': 0.005,
                'max_rel_err': 0.01,
                'mean_rel_err': 0.003,
                'eb': 0.002
            },
            'gpu': {
                'small_value_error_rate': 0.0009,
                'rmse': 0.0048,
                'max_rel_err': 0.009,
                'mean_rel_err': 0.0028,
                'eb': 0.0019
            }
        }
        self.compare = BenchmarkPrecisionCompare(self.input_data)

    def test_get_final_status(self):
        # 测试所有通过的情况
        status_list = [CompareConst.PASS, CompareConst.PASS, CompareConst.PASS]
        result = self.compare.get_final_status(status_list)
        self.assertEqual(result, CompareConst.PASS)

        # 测试有WARNING的情况
        status_list = [CompareConst.PASS, CompareConst.WARNING, CompareConst.PASS]
        result = self.compare.get_final_status(status_list)
        self.assertEqual(result, CompareConst.WARNING)

        # 测试有ERROR的情况
        status_list = [CompareConst.PASS, CompareConst.ERROR, CompareConst.WARNING]
        result = self.compare.get_final_status(status_list)
        self.assertEqual(result, CompareConst.ERROR)

    def test_calc_ratio(self):
        # 测试正常数值的情况
        ratio, consistency, message = self.compare._calc_ratio('small_value_error_rate')
        self.assertIsInstance(ratio, float)
        self.assertTrue(consistency)
        self.assertEqual(message, "")

        # 测试inf/nan的情况
        input_data_with_nan = {
            'npu': {'small_value_error_rate': float('nan')},
            'gpu': {'small_value_error_rate': 0.001}
        }
        compare_nan = BenchmarkPrecisionCompare(input_data_with_nan)
        ratio, consistency, message = compare_nan._calc_ratio('small_value_error_rate')
        self.assertFalse(consistency)
        self.assertNotEqual(message, "")

    def test_compute_ratio(self):
        metrics, inf_nan_consistency = self.compare._compute_ratio()
        
        # 验证返回的metrics字典包含所有必要的键
        expected_keys = [
            "small_value_err_ratio",
            "rmse_ratio",
            "max_rel_err_ratio",
            "mean_rel_err_ratio",
            "eb_ratio",
            "compare_message"
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)
            
        # 验证inf_nan_consistency的所有属性都是布尔值
        self.assertTrue(all(isinstance(x, bool) for x in inf_nan_consistency))

    def test_get_single_metric_status(self):
        # 测试不同阈值的情况
        # PASS的情况
        status = self.compare._get_single_metric_status(1.1, CompareConst.RMSE)
        self.assertEqual(status, CompareConst.PASS)

        # WARNING的情况
        status = self.compare._get_single_metric_status(5.0, CompareConst.RMSE)
        self.assertEqual(status, CompareConst.WARNING)

        # ERROR的情况
        status = self.compare._get_single_metric_status(10.0, CompareConst.RMSE)
        self.assertEqual(status, CompareConst.ERROR)

        # inf/nan的情况
        status = self.compare._get_single_metric_status(float('inf'), CompareConst.RMSE)
        self.assertEqual(status, CompareConst.PASS)

    def test_get_status(self):
        # 准备测试数据
        metrics = {
            "small_value_err_ratio": 1.1,
            "rmse_ratio": 1.2,
            "max_rel_err_ratio": 1.3,
            "mean_rel_err_ratio": 1.4,
            "eb_ratio": 1.5,
            "compare_message": ""
        }
        
        inf_nan_consistency = BenchmarkInfNanConsistency(
            small_value_inf_nan_consistency=True,
            rmse_inf_nan_consistency=True,
            max_rel_inf_nan_consistency=True,
            mean_rel_inf_nan_consistency=True,
            eb_inf_nan_consistency=True
        )

        result = self.compare._get_status(metrics, inf_nan_consistency)
        
        # 验证返回的结果包含所有必要的状态
        expected_status_keys = [
            CompareConst.SMALL_VALUE_ERR_STATUS,
            CompareConst.RMSE_STATUS,
            CompareConst.MAX_REL_ERR_STATUS,
            CompareConst.MEAN_REL_ERR_STATUS,
            CompareConst.EB_STATUS,
            CompareConst.COMPARE_RESULT
        ]
        for key in expected_status_keys:
            self.assertIn(key, result)

if __name__ == '__main__':
    unittest.main()