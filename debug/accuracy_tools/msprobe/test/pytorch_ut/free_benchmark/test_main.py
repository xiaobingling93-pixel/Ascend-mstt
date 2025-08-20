from abc import ABC
from unittest import TestCase

import torch
import torch.nn as nn
from msprobe.core.common.const import Const
from msprobe.pytorch.free_benchmark import FreeBenchmarkCheck
from msprobe.pytorch.free_benchmark.common.constant import CommonField, PreheatConfig
from msprobe.pytorch.free_benchmark.common.enums import (
    DeviceType,
    FuzzLevel,
    HandlerType,
    PerturbationMode,
)
from msprobe.pytorch.free_benchmark.result_handlers.handler_factory import (
    FuzzHandlerFactory,
)


class Config(ABC):
    """
    用以提供参数配置
    """

    def __init__(self, fuzz_stage, handler_type):
        self.fuzz_stage = fuzz_stage
        self.handler_type = handler_type
        self.fuzz_device = DeviceType.NPU
        self.fuzz_level = FuzzLevel.BASE_LEVEL
        self.pert_mode = PerturbationMode.IMPROVE_PRECISION
        self.preheat_config = {PreheatConfig.IF_PREHEAT: False}


class WrapMul(nn.Module):
    """
    用nn.module包装mul算子, 在forward中调用torch.mul
    """

    def __init__(self, op_name) -> None:
        super().__init__()
        self.op_name = op_name

    def forward(self, *args, **kwargs):
        return torch.mul(*args, **kwargs)


class UnequalDataProcessor(ABC):
    """
    接口类, 处理检测不一致结果
    """

    def __init__(self) -> None:
        super().__init__()
        self.unequal_rows = []

    def update_unequal_rows(self, unequal_rows):
        if unequal_rows:
            self.unequal_rows.append(unequal_rows)


class TestInterface(TestCase):
    def setUp(self):
        self.api_name = "Torch.mul.0"

    def test_init_with_none(self):
        # 对于全为none的输入初始化无标杆实例，检查其默认值
        config = Config(None, None)
        config.pert_mode = None
        config.fuzz_level = None
        config.fuzz_device = None
        checker = FreeBenchmarkCheck(config)
        self.assertEqual(checker.config.pert_mode, PerturbationMode.IMPROVE_PRECISION)
        self.assertEqual(checker.config.fuzz_level, FuzzLevel.BASE_LEVEL)
        self.assertEqual(checker.config.fuzz_device, DeviceType.NPU)

    def testForwardFix(self):
        # 对于前向接口，在forward钩子中开启FIX，返回结果给hook的输出
        # 为了与下一层的输入对齐、应该转换为扰动前输出的dtype，否则可能报错
        config = Config(Const.FORWARD, HandlerType.FIX)
        checker = FreeBenchmarkCheck(config)
        # 执行算子前向
        x = torch.randn(2, 3).to(torch.float16)
        y = torch.randn(2, 3).to(torch.float16)
        mul_module = WrapMul(self.api_name)
        out = mul_module(x, y)
        # 模拟forward hook中调用无标杆前向检测接口
        result, _ = checker.forward(
            self.api_name,
            mul_module,
            args=(x, y),
            kwargs={},
            output=out,
        )
        self.assertEqual(result.dtype, torch.float16)
