import functools
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

    def testBackwardCheck(self):
        # 对于反向接口，在pre forward时暂存input, 然后在backwrad后进行对比
        config = Config(Const.BACKWARD, HandlerType.CHECK)
        checker = FreeBenchmarkCheck(config)
        processor = UnequalDataProcessor()
        # 初始化输入输出
        x = torch.tensor([2, 3], dtype=torch.float16, requires_grad=True)
        y = torch.tensor([2, 3], dtype=torch.float16, requires_grad=True)
        grad_output = torch.tensor([1, 1], dtype=torch.float16)
        backward_name = Const.SEP.join([self.api_name, Const.BACKWARD])
        # 执行前向生成grad saver实例
        mul_module = WrapMul(self.api_name)
        checker.pre_forward(backward_name, mul_module, processor, (x, y), {})
        # 执行算子前向和反向, 并反向获取扰动后grad_input
        out = mul_module(x, y)
        checker.backward(backward_name, mul_module, grad_output)
        out.backward(torch.ones_like(out))
        # module是否添加暂存器, 其中反向钩子执行扰动后grad_input是否正确
        self.assertTrue(hasattr(mul_module, CommonField.GRADSAVER))
        grad_saver = getattr(mul_module, CommonField.GRADSAVER)
        self.assertEqual(grad_saver.perturbed_grad_input[0][0], 2)
        handler = FuzzHandlerFactory.create(grad_saver.handler_params)
        # 模拟一个张量的梯度更新时触发反向检测
        grad_saver.compare_grad_results(
            handler, torch.tensor(1.0), torch.tensor(2.0), 0
        )
        self.assertEqual(len(processor.unequal_rows), 0)
