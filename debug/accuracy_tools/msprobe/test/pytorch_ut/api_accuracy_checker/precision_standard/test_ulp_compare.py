import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch

from msprobe.pytorch.api_accuracy_checker.precision_standard.ulp_compare import UlpPrecisionCompare
from msprobe.core.common.const import CompareConst
from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import ApiPrecisionCompareColumn

class TestUlpPrecisionCompare(unittest.TestCase):
    def setUp(self):
        self.mock_input_data = {
            'row_npu': {
                ApiPrecisionCompareColumn.MEAN_ULP_ERR: 1.0,
                ApiPrecisionCompareColumn.ULP_ERR_PROPORTION: 0.1,
                ApiPrecisionCompareColumn.DEVICE_DTYPE: 'torch.float32'
            },
            'row_gpu': {
                ApiPrecisionCompareColumn.MEAN_ULP_ERR: 1.0,
                ApiPrecisionCompareColumn.ULP_ERR_PROPORTION: 0.1
            }
        }
        self.compare = UlpPrecisionCompare(self.mock_input_data)

    def test_compute_normal_values(self):
        """测试正常数值的计算"""
        metrics, inf_nan_consistency = self.compare._compute_ratio()
        
        self.assertEqual(metrics[CompareConst.MEAN_ULP_ERR], 1.0)
        self.assertEqual(metrics[CompareConst.ULP_ERR_PROPORTION], 0.1)
        self.assertEqual(metrics[CompareConst.ULP_ERR_PROPORTION_RATIO], 1.0)
        self.assertTrue(inf_nan_consistency.mean_ulp_err_inf_nan_consistency)
        self.assertTrue(inf_nan_consistency.ulp_err_proportion_ratio_inf_nan_consistency)

    def test_compute_with_inf(self):
        """测试包含inf值的计算"""
        self.mock_input_data['row_npu'][ApiPrecisionCompareColumn.MEAN_ULP_ERR] = float('inf')
        self.mock_input_data['row_gpu'][ApiPrecisionCompareColumn.MEAN_ULP_ERR] = float('inf')
        compare = UlpPrecisionCompare(self.mock_input_data)
        
        metrics, inf_nan_consistency = compare._compute_ratio()
        self.assertTrue(inf_nan_consistency.mean_ulp_err_inf_nan_consistency)

    def test_fp32_pass_criteria(self):
        """测试fp32数据类型的通过条件"""
        with patch('msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config.StandardConfig.get_ulp_threshold') as mock_threshold:
            mock_threshold.return_value = (2.0, 0.2, 2.0)  # 设置阈值
            
            metrics = {
                CompareConst.MEAN_ULP_ERR: 1.0,
                CompareConst.ULP_ERR_PROPORTION: 0.1,
                CompareConst.ULP_ERR_PROPORTION_RATIO: 1.0,
                CompareConst.COMPARE_MESSAGE: ""
            }
            
            result = self.compare._get_status(metrics, Mock(mean_ulp_err_inf_nan_consistency=True,
                                                          ulp_err_proportion_ratio_inf_nan_consistency=True))
            
            self.assertEqual(result[CompareConst.COMPARE_RESULT], CompareConst.PASS)

    def test_fp16_pass_criteria(self):
        """测试fp16数据类型的通过条件"""
        self.mock_input_data['row_npu'][ApiPrecisionCompareColumn.DEVICE_DTYPE] = 'torch.float16'
        compare = UlpPrecisionCompare(self.mock_input_data)
        
        with patch('msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config.StandardConfig.get_ulp_threshold') as mock_threshold:
            mock_threshold.return_value = (None, 0.2, 2.0)  # 设置fp16阈值
            
            metrics = {
                CompareConst.ULP_ERR_PROPORTION: 0.1,
                CompareConst.ULP_ERR_PROPORTION_RATIO: 1.0,
                CompareConst.COMPARE_MESSAGE: ""
            }
            
            result = compare._get_status(metrics, Mock(mean_ulp_err_inf_nan_consistency=True,
                                                     ulp_err_proportion_ratio_inf_nan_consistency=True))
            
            self.assertEqual(result[CompareConst.COMPARE_RESULT], CompareConst.PASS)

    def test_fail_criteria(self):
        """测试不满足精度要求的情况"""
        with patch('msprobe.pytorch.api_accuracy_checker.precision_standard.standard_config.StandardConfig.get_ulp_threshold') as mock_threshold:
            mock_threshold.return_value = (0.5, 0.05, 1.0)  # 设置较严格的阈值
            
            metrics = {
                CompareConst.MEAN_ULP_ERR: 1.0,
                CompareConst.ULP_ERR_PROPORTION: 0.1,
                CompareConst.ULP_ERR_PROPORTION_RATIO: 1.5,
                CompareConst.COMPARE_MESSAGE: ""
            }
            
            result = self.compare._get_status(metrics, Mock(mean_ulp_err_inf_nan_consistency=True,
                                                          ulp_err_proportion_ratio_inf_nan_consistency=True))
            
            self.assertEqual(result[CompareConst.COMPARE_RESULT], CompareConst.ERROR)
            self.assertIn("ULP误差不满足标准", result[CompareConst.COMPARE_MESSAGE])