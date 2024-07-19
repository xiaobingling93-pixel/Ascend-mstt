from abc import ABC
from unittest import TestCase

import torch
from msprobe.core.common.const import Const
from msprobe.pytorch.free_benchmark.common.constant import PreheatConfig, ThresholdConfig
from msprobe.pytorch.free_benchmark.common.counter import preheat_counter
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
    def __init__(self, handler_type, preheat_config):
        self.fuzz_stage = Const.FORWARD
        self.handler_type = handler_type
        self.fuzz_device = DeviceType.NPU
        self.fuzz_level = FuzzLevel.BASE_LEVEL
        self.pert_mode = PerturbationMode.IMPROVE_PRECISION
        self.preheat_config = preheat_config


class TestFuzzHandler(TestCase):

    def setUp(self) -> None:
        origin_inputs = [
            torch.as_tensor([3.01, 3.02], dtype=torch.float16),
            torch.as_tensor([0.02, 0.02], dtype=torch.float16),
        ]
        # 将输入乘以一个大于误差阈值1.002的值，模拟二次执行出现误差
        perturbed_inputs = [
            (value * 1.0021).to(torch.float32).to("cpu") for value in origin_inputs
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
        self.step = 0

    def test_result_handler_check(self):
        # 对于check处理类，扰动前后输出不一致的情况会有UnequalRow对象生成
        for _ in range(2):
            config = Config(
                HandlerType.CHECK, {PreheatConfig.IF_PREHEAT: False}
            )
            handler_params = make_handler_params(self.api_name, config, self.step)
            handler = FuzzHandlerFactory.create(handler_params)
            handler.handle(self.data_params)
            self.assertEqual(
                len(handler.get_unequal_rows()), 1
            )

    def test_result_handler_fix(self):
        # 对于fix处理类，扰动后输出会替代原始输出, dtype和原始输出一致，但值为新输出值
        config = Config(
            HandlerType.FIX, {PreheatConfig.IF_PREHEAT: False}
        )
        handler_params = make_handler_params(self.api_name, config, self.step)
        handler = FuzzHandlerFactory.create(handler_params)
        result = handler.handle(self.data_params)
        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.device, self.data_params.original_result.device)
        self.assertAlmostEqual(
            result[0], self.data_params.perturbed_result.to(torch.float16)[0]
        )
        self.assertAlmostEqual(
            result[1], self.data_params.perturbed_result.to(torch.float16)[1]
        )

    def test_result_handler_preheat(self):
        # 对于preheat处理类，在预热阶段后的阈值会根据CPU调整
        config = Config(
            HandlerType.CHECK,
            {
                PreheatConfig.IF_PREHEAT: True,
                PreheatConfig.PREHEAT_STEP: 4,
                PreheatConfig.MAX_SAMPLE: 3
            }
        )
        for _ in range(3):
            handler_params = make_handler_params(self.api_name, config, 0)
            handler = FuzzHandlerFactory.create(handler_params)
            handler.handle(self.data_params)
        # 通过preheat_counter的数据可以判断预热是否正常执行，这里第一个step会记录api执行次数
        self.assertEqual(preheat_counter.get_one_step_used_api("add"), 3)
        for step in range(1, 4):
            for _ in range(3):
                handler_params = make_handler_params(self.api_name, config, step)
                handler = FuzzHandlerFactory.create(handler_params)
                handler.handle(self.data_params)
            # call time记录当前step api的调用次数
            self.assertEqual(preheat_counter.get_api_called_time("add"), 3)
            # 对于3个step最多采样三次的预热设置，sample time应该每次采样一例
            self.assertEqual(preheat_counter.get_api_sample_time("add"), 1)
            # 预热阶段，api阈值应该在两个阈值超参之间
            api_threshld = preheat_counter.get_api_thd("add", "torch.float16")
            self.assertLessEqual(
                api_threshld,
                ThresholdConfig.PREHEAT_INITIAL_THD
            )
            self.assertGreaterEqual(
                api_threshld,
                ThresholdConfig.DTYPE_PER_THD[torch.float16]
            )
