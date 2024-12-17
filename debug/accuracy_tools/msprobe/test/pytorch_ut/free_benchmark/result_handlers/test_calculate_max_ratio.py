from unittest import TestCase

import torch
from msprobe.pytorch.free_benchmark.common.constant import ThresholdConfig
from msprobe.pytorch.free_benchmark.common.params import HandlerParams
from msprobe.pytorch.free_benchmark.result_handlers.check_handler import CheckerHandler


class TestFuzzHandler(TestCase):

    def setUp(self) -> None:
        self.api_name = "test_api"
        self.handler = CheckerHandler(HandlerParams(api_name=self.api_name))
        self.abs_tol = 1e-4

    def test_calculate_max_ratio_with_equal_outputs(self):
        # 测试两个输出相等时，比值应该接近1
        origin_output = torch.tensor([1.0, 2.0, 3.0])
        perturbed_output = torch.tensor([1.0, 2.0, 3.0])
        max_ratio = self.handler.calculate_max_ratio(
            origin_output, perturbed_output, self.abs_tol
        )
        self.assertAlmostEqual(max_ratio, 1.0)

    def test_calculate_max_ratio_with_different_outputs(self):
        # 测试两个输出不同时，比值应该为最大的比值
        origin_output = torch.tensor([1.0, 2.0, 1e-4])
        perturbed_output = torch.tensor([1.3, 2.7, 1e-3])
        max_ratio = self.handler.calculate_max_ratio(
            origin_output, perturbed_output, self.abs_tol
        )
        self.assertAlmostEqual(max_ratio, 10.0, places=2)

    def test_calculate_max_ratio_with_tol_elements(self):
        # 测试忽略绝对值小于极小值的情况，小于的全部变为极小值计算
        origin_output = torch.tensor([1.0, 1e-8, 1e-6])
        perturbed_output = torch.tensor([1.0, 1e-4, -1e-8])
        max_ratio = self.handler.calculate_max_ratio(
            origin_output, perturbed_output, self.abs_tol
        )
        self.assertAlmostEqual(max_ratio, 1.0)

    def test_calculate_max_ratio_with_symbol_flipping(self):
        # 测试乘积符号相反时，应该返回SYMBOL_FLIPPING
        origin_output = torch.tensor([1.0, -2.0, 3.0])
        perturbed_output = torch.tensor([1.0, 2.0, 3.0])
        result = self.handler.calculate_max_ratio(
            origin_output, perturbed_output, self.abs_tol
        )
        self.assertEqual(result, ThresholdConfig.SYMBOL_FLIPPING)

    def test_calculate_max_ratio_with_nan_values(self):
        # 测试包含NaN值时，函数应该正确计算
        origin_output = torch.tensor([1.0, float("nan"), 2.0])
        perturbed_output = torch.tensor([1.1, float("nan"), 2.4])
        max_ratio = self.handler.calculate_max_ratio(
            origin_output, perturbed_output, self.abs_tol
        )
        self.assertAlmostEqual(max_ratio, 1.2)

    def test_calculate_max_ratio_with_empty_chunks(self):
        # 测试空的输出块时，函数应该正确处理
        origin_output = torch.tensor([])
        perturbed_output = torch.tensor([])
        max_ratio = self.handler.calculate_max_ratio(
            origin_output, perturbed_output, self.abs_tol
        )
        self.assertEqual(max_ratio, ThresholdConfig.COMP_CONSISTENT)
