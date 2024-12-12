import unittest
import torch
import numpy as np

from msprobe.pytorch.api_accuracy_checker.precision_standard.ulp_compare import UlpPrecisionCompare
from msprobe.core.common.const import CompareConst
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import ApiPrecisionCompareColumn
from msprobe.pytorch.api_accuracy_checker.compare.compare_input import PrecisionCompareInput
from msprobe.pytorch.api_accuracy_checker.compare.compare_column import ApiPrecisionOutputColumn

class TestUlpPrecisionCompare(unittest.TestCase):
    def setUp(self):
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

        compare_column = ApiPrecisionOutputColumn()
        input_data = PrecisionCompareInput(self.npu_precision, self.gpu_precision, compare_column)
        self.ulp_standard = UlpPrecisionCompare(input_data)

    def test_init(self):
        """测试初始化函数"""
        self.assertEqual(self.ulp_standard.compare_algorithm, CompareConst.ULP_COMPARE_ALGORITHM_NAME)

    def test_compute_ulp_err_proportion_ratio(self):
        """测试_compute_ulp_err_proportion_ratio函数"""
        # 测试正常数值
        ratio, consistency, message = self.ulp_standard._compute_ulp_err_proportion_ratio(0.1, 0.08)
        self.assertIsInstance(ratio, float)
        self.assertTrue(consistency)
        self.assertEqual(message, "")

        # 测试一个值为inf
        ratio, consistency, message = self.ulp_standard._compute_ulp_err_proportion_ratio(float('inf'), 0.08)
        self.assertFalse(consistency)
        self.assertNotEqual(message, "")

        # 测试两个值都为inf
        ratio, consistency, message = self.ulp_standard._compute_ulp_err_proportion_ratio(float('inf'), float('inf'))
        self.assertTrue(consistency)
        self.assertEqual(message, "ULP误差大于阈值占比同为同号inf或nan\n")

    def test_compute_mean_ulp_err(self):
        """测试_compute_mean_ulp_err函数"""
        # 测试正常数值
        mean_ulp_err, consistency, message = self.ulp_standard._compute_mean_ulp_err()
        self.assertEqual(mean_ulp_err, 0.1)
        self.assertTrue(consistency)
        self.assertEqual(message, "")

        # 测试NPU值为nan
        self.npu_precision[ApiPrecisionCompareColumn.MEAN_ULP_ERR] = float('nan')
        mean_ulp_err, consistency, message = self.ulp_standard._compute_mean_ulp_err()
        self.assertTrue(np.isnan(mean_ulp_err))
        self.assertFalse(consistency)
        self.assertNotEqual(message, "")

    def test_compute_ulp_err_proportion(self):
        """测试_compute_ulp_err_proportion函数"""
        npu_value, gpu_value = self.ulp_standard._compute_ulp_err_proportion()
        self.assertEqual(npu_value, 0.05)
        self.assertEqual(gpu_value, 0.06)

    def test_get_fp32_ulp_err_status(self):
        """测试_get_fp32_ulp_err_status函数"""
        # 测试mean_ulp_err通过的情况
        status, message = self.ulp_standard._get_fp32_ulp_err_status(0.1, 0.1, 1.0)
        self.assertEqual(status, CompareConst.PASS)
        self.assertEqual(message, "")

        # 测试ulp_err_proportion通过的情况
        status, message = self.ulp_standard._get_fp32_ulp_err_status(64, 0.04, 0.9)
        self.assertEqual(status, CompareConst.PASS)
        self.assertEqual(message, "")

        # 测试都不通过的情况
        status, message = self.ulp_standard._get_fp32_ulp_err_status(64, 0.05, 1.0)
        self.assertEqual(status, CompareConst.ERROR)
        self.assertIn("ULP误差不满足标准", message)

    def test_get_fp16_ulp_err_status(self):
        """测试_get_fp16_ulp_err_status函数"""
        # 测试ulp_err_proportion通过的情况
        status, message = self.ulp_standard._get_fp16_ulp_err_status(0.0001, 0.1)
        self.assertEqual(status, CompareConst.PASS)
        self.assertEqual(message, "")

        # 测试ulp_err_proportion_ratio通过的情况
        status, message = self.ulp_standard._get_fp16_ulp_err_status(0.2, 0.1)
        self.assertEqual(status, CompareConst.PASS)
        self.assertEqual(message, "")

        # 测试都不通过的情况
        status, message = self.ulp_standard._get_fp16_ulp_err_status(1.0, 2.0)
        self.assertEqual(status, CompareConst.ERROR)
        self.assertIn("ULP误差不满足标准", message)

    def test_compute_ratio(self):
        """测试_compute_ratio函数"""
        metrics, inf_nan_consistency = self.ulp_standard._compute_ratio()
        
        # 验证返回的metrics包含所有必要的键
        self.assertIn(CompareConst.MEAN_ULP_ERR, metrics)
        self.assertIn(CompareConst.ULP_ERR_PROPORTION, metrics)
        self.assertIn(CompareConst.ULP_ERR_PROPORTION_RATIO, metrics)
        self.assertIn(CompareConst.COMPARE_MESSAGE, metrics)
        
        # 验证inf_nan_consistency的类型和属性
        self.assertTrue(hasattr(inf_nan_consistency, 'mean_ulp_err_inf_nan_consistency'))
        self.assertTrue(hasattr(inf_nan_consistency, 'ulp_err_proportion_ratio_inf_nan_consistency'))

    def test_get_status_with_inf_nan_inconsistency(self):
        """测试_get_status函数在inf/nan不一致的情况"""
        metrics = {
            CompareConst.MEAN_ULP_ERR: 0.5,
            CompareConst.ULP_ERR_PROPORTION: 0.1,
            CompareConst.ULP_ERR_PROPORTION_RATIO: 1.0,
            CompareConst.COMPARE_MESSAGE: ""
        }
        
        inf_nan_consistency = type('UlpInfNanConsistency', (), {
            'mean_ulp_err_inf_nan_consistency': False,
            'ulp_err_proportion_ratio_inf_nan_consistency': True
        })
        
        result = self.ulp_standard._get_status(metrics, inf_nan_consistency)
        self.assertEqual(result[CompareConst.COMPARE_RESULT], CompareConst.ERROR)
        self.assertIn("ULP误差不满足标准", result[CompareConst.COMPARE_MESSAGE])

    def test_get_status_with_different_dtypes(self):
        """测试_get_status函数对不同数据类型的处理"""
        metrics = {
            CompareConst.MEAN_ULP_ERR: 0.5,
            CompareConst.ULP_ERR_PROPORTION: 0.001,
            CompareConst.ULP_ERR_PROPORTION_RATIO: 0.9,
            CompareConst.COMPARE_MESSAGE: ""
        }
        
        inf_nan_consistency = type('UlpInfNanConsistency', (), {
            'mean_ulp_err_inf_nan_consistency': True,
            'ulp_err_proportion_ratio_inf_nan_consistency': True
        })

        # 测试fp32
        self.npu_precision[ApiPrecisionCompareColumn.DEVICE_DTYPE] = 'torch.float32'
        result = self.ulp_standard._get_status(metrics, inf_nan_consistency)
        self.assertEqual(result[CompareConst.COMPARE_RESULT], CompareConst.PASS)

        # 测试fp16
        self.npu_precision[ApiPrecisionCompareColumn.DEVICE_DTYPE] = 'torch.float16'
        result = self.ulp_standard._get_status(metrics, inf_nan_consistency)
        self.assertEqual(result[CompareConst.COMPARE_RESULT], CompareConst.PASS)

if __name__ == '__main__':
    unittest.main()