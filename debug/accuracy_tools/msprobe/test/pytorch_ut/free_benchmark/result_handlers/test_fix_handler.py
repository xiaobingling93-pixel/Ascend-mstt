from abc import ABC
from unittest import TestCase
from unittest.mock import patch

import torch
from msprobe.core.common.const import Const
from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.constant import (
    PreheatConfig,
)
from msprobe.pytorch.free_benchmark.common.enums import (
    DeviceType,
    FuzzLevel,
    HandlerType,
    PerturbationMode,
)
from msprobe.pytorch.free_benchmark.common.params import DataParams, make_handler_params
from msprobe.pytorch.free_benchmark.result_handlers.handler_factory import (
    FuzzHandlerFactory,
)


class Config(ABC):
    """
    用以提供参数配置
    """

    def __init__(self):
        self.fuzz_stage = Const.FORWARD
        self.handler_type = HandlerType.FIX
        self.fuzz_device = DeviceType.NPU
        self.fuzz_level = FuzzLevel.BASE_LEVEL
        self.pert_mode = PerturbationMode.IMPROVE_PRECISION
        self.preheat_config = {PreheatConfig.IF_PREHEAT: False}


class TestFuzzHandler(TestCase):

    def setUp(self) -> None:
        origin_inputs = [
            torch.as_tensor([3.01, 3.02], dtype=torch.float16),
            torch.as_tensor([0.02, 0.02], dtype=torch.float16),
        ]
        # 将输入乘以一个大于误差阈值1.004的值，模拟二次执行出现误差
        perturbed_inputs = [
            (value * 1.0041).to(torch.float32).to("cpu") for value in origin_inputs
        ]
        origin_output = torch.add(*origin_inputs)
        perturbed_output = torch.add(*perturbed_inputs)
        # 实例有问题的data对象
        self.data_params = DataParams(
            args=origin_inputs,
            kwargs={},
            original_result=origin_output,
            perturbed_result=perturbed_output,
            origin_func=torch.add,
        )
        self.api_name = "add.0.forward"
        handler_params = make_handler_params(self.api_name, Config(), 0)
        self.handler = FuzzHandlerFactory.create(handler_params)

    def test_result_handler_fix(self):
        # 对于fix处理类，扰动后输出会替代原始输出, dtype和原始输出一致，但值为新输出值
        self.data_params.original_result = [
            self.data_params.original_result,
            {"res": self.data_params.original_result},
        ]
        self.data_params.perturbed_result = [
            self.data_params.perturbed_result,
            {"res": self.data_params.perturbed_result},
        ]
        result = self.handler.handle(self.data_params)
        self.assertEqual(result[0].dtype, torch.float16)
        self.assertEqual(result[0].device, self.data_params.original_result[0].device)
        self.assertAlmostEqual(
            result[1]["res"][0],
            self.data_params.perturbed_result[1]["res"].to(torch.float16)[0],
        )
        self.assertAlmostEqual(
            result[1]["res"][1],
            self.data_params.perturbed_result[1]["res"].to(torch.float16)[1],
        )

    @patch.object(logger, "warning")
    def test_fix_handler_with_index_error(self, mock_logger):
        # 对于fix处理类，扰动后输出替代原始输出遇到indexerror会打印warning
        self.data_params.original_result = [
            self.data_params.original_result,
            # 原始输出多出一个张量
            torch.as_tensor(3, dtype=torch.float16),
        ]
        self.data_params.perturbed_result = [
            self.data_params.perturbed_result,
        ]
        self.handler.handle(self.data_params)
        mock_logger.assert_called_with(
            f"[msprobe] Free Benchmark: For {self.api_name} "
            f"Fix output failed because of: \n"
            f"[msprobe] Free benchmark invalid perturbed output: "
            f"length of perturbed output (1) is different from the length of original output (2)."
        )

    @patch.object(logger, "warning")
    def test_fix_handler_with_key_error(self, mock_logger):
        # 对于fix处理类，扰动后输出替代原始输出遇到KeyError会打印warning
        self.data_params.original_result = {"res": self.data_params.original_result}
        self.data_params.perturbed_result = {"res_err": self.data_params.perturbed_result}

        self.handler.handle(self.data_params)
        mock_logger.assert_called_with(
            f"[msprobe] Free Benchmark: For {self.api_name} "
            f"Fix output failed because of: \n"
            f"[msprobe] Free benchmark invalid perturbed output: "
            f"'res' not in perturbed output."
        )

    @patch.object(logger, "warning")
    def test_fix_handler_with_type_error(self, mock_logger):
        # 对于fix处理类，扰动后输出替代原始输出遇到输入输出类型不匹配，会打印warning
        self.data_params.original_result = [self.data_params.original_result]
        self.data_params.perturbed_result = {"res": self.data_params.perturbed_result}
        self.handler.handle(self.data_params)
        mock_logger.assert_called_with(
            f"[msprobe] Free Benchmark: For {self.api_name} "
            f"Fix output failed because of: \n"
            f"[msprobe] Free benchmark get unsupported type: conversion of two outputs"
            f" with types ({type([])}, {type({})}) is not supported."
        )
