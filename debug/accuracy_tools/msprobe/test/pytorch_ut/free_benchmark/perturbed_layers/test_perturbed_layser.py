from unittest import TestCase
from unittest.mock import patch

import torch
from msprobe.core.common.const import Const
from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.enums import DeviceType, PerturbationMode
from msprobe.pytorch.free_benchmark.common.params import data_pre_deal
from msprobe.pytorch.free_benchmark.perturbed_layers.layer_factory import LayerFactory


class TestPerturbedLayer(TestCase):

    # 对输出精度和输入精度一致算子使用升精度扰动因子时, 输出结果的精度也会提升
    def test_improve_precision_layer_handle_with_out_dtype_changing(self):
        api_name = "Torch.mul.0.forward"
        x = torch.randn(2, 3, dtype=torch.float16)
        y = torch.randn(2, 3, dtype=torch.float16)
        out = torch.mul(x, y)

        data_params = data_pre_deal(api_name, torch.mul, (x, y), {})
        data_params.fuzz_stage = Const.FORWARD
        data_params.original_result = out

        layer = LayerFactory.create(
            api_name, DeviceType.NPU, PerturbationMode.IMPROVE_PRECISION
        )
        layer.handle(data_params)
        self.assertEqual(data_params.original_result.dtype, torch.float16)
        self.assertEqual(layer.perturbed_value, torch.float32)
        self.assertEqual(data_params.perturbed_result.dtype, torch.float32)

    # 对于可迭代类型的输入, 升精度方法会遍历其中元素对支持类型输入升精度
    def test_improve_precision_layer_with_iterable_inputs(self):
        api_name = "iterable.0.forward"
        tensor_a = torch.randn(2, 3, dtype=torch.bfloat16)
        tensor_b = torch.randn(2, 3, dtype=torch.float16)
        tensor_c = torch.randn(2, 3, dtype=torch.float32)
        tensor_d = torch.randn(2, 3, dtype=torch.float64)
        tensor_f = torch.randn(2, 3, dtype=torch.float64).to(torch.int32)
        inputs = [tensor_a, tensor_b, {"c": tensor_c, "d": tensor_d}, tensor_f]

        layer = LayerFactory.create(
            api_name, DeviceType.NPU, PerturbationMode.IMPROVE_PRECISION
        )
        Perturbed_value = layer.improve_tensor_precision(inputs)
        self.assertEqual(Perturbed_value[0].dtype, torch.float32)
        self.assertEqual(Perturbed_value[1].dtype, torch.float32)
        self.assertEqual(Perturbed_value[2]["c"].dtype, torch.float32)
        self.assertEqual(Perturbed_value[2]["d"].dtype, torch.float64)
        self.assertEqual(Perturbed_value[3].dtype, torch.int32)

    # no_change扰动因子不会改变输入输出
    def test_no_change_layer(self):
        api_name = "Torch.mul.0.forward"
        x = torch.randn(2, 3, dtype=torch.float16)
        y = torch.randn(2, 3, dtype=torch.float16)
        out = torch.Tensor.mul(x, y)
        check_element = torch.clone(out[0][0])
        data_params = data_pre_deal(api_name, torch.Tensor.mul, (x, y), {})
        data_params.fuzz_stage = Const.FORWARD
        data_params.original_result = out

        layer = LayerFactory.create(
            api_name, DeviceType.NPU, PerturbationMode.NO_CHANGE
        )
        layer.handle(data_params)
        self.assertEqual(check_element, out[0][0])
        self.assertEqual(data_params.perturbed_result[0].dtype, torch.float16)

    # 对于一维二维张量，change_value扰动因子会交换首尾值的位置
    def test_change_value_layer(self):
        api_name = "change.0.forward"
        inputs_1dim = torch.as_tensor([1e-9, 1e-7, 1e-2], dtype=torch.float32)
        inputs_2dim = torch.as_tensor(
            [[1e-9, 1e-7, 1e-2], [1e-9, 1e-2, 1e-7]], dtype=torch.float32
        )
        layer = LayerFactory.create(
            api_name, DeviceType.NPU, PerturbationMode.CHANGE_VALUE
        )
        Perturbed_value_1dim = layer.change_value(inputs_1dim)
        layer.is_added = False
        Perturbed_value_2dim = layer.change_value(inputs_2dim)
        self.assertEqual(Perturbed_value_1dim[0], 1e-2)
        self.assertEqual(Perturbed_value_1dim[2], 1e-9)
        self.assertEqual(Perturbed_value_2dim[0][0], 1e-7)
        self.assertEqual(Perturbed_value_2dim[-1][-1], 1e-9)

    # 对于可迭代类型的输入, change_value扰动会遍历其中元素，但只会交换第一个张量首尾值
    def test_change_value_layer_with_iterable_inputs(self):
        api_name = "iterable.0.forward"
        tensor_a = torch.as_tensor([1e-9, 1e-7, 1e-2], dtype=torch.bfloat16)
        inputs = [tensor_a, tensor_a, {"b": tensor_a}]
        layer = LayerFactory.create(
            api_name, DeviceType.NPU, PerturbationMode.CHANGE_VALUE
        )
        Perturbed_value = layer.change_value(inputs)
        self.assertEqual(Perturbed_value[0][0], 1e-2)
        self.assertEqual(Perturbed_value[0][2], 1e-9)
        self.assertEqual(Perturbed_value[1][0], 1e-9)
        self.assertEqual(Perturbed_value[1][-1], 1e-2)
        self.assertEqual(Perturbed_value[2]["b"][0], 1e-9)
        self.assertEqual(Perturbed_value[2]["b"][-1], 1e-2)

    # 对于元素个数不符合要求的，change_value会打印告警日志
    @patch.object(logger, "info_on_rank_0")
    def test_change_value_layer_with_invalid_tensor_size(self, mock_logger):
        x = torch.as_tensor([[], []])
        y = torch.as_tensor([])
        layer = LayerFactory.create(
            "test_api_name", DeviceType.NPU, PerturbationMode.CHANGE_VALUE
        )
        layer.pre_check(x)
        mock_logger.assert_called_with(
            "[msprobe] Free Benchmark: For test_api_name with ndim 2, "
            "at least one of 0-dimension or 1-dimension greater than 1. Cancel change value."
        )
        layer.pre_check(y)
        mock_logger.assert_called_with(
            "[msprobe] Free Benchmark: For test_api_name, "
            "0-dimension must greater than 1. Cancel change value."
        )

    # 对于输入张量，bit_noise扰动因子对大于极小值的部分进行末尾比特翻转
    def test_bit_noise_layer(self):
        api_name = "bitnoise.0.forward"
        inputs = torch.as_tensor(
            [4096.00048828125, 16777216, 1e-38], dtype=torch.float32
        )
        layer = LayerFactory.create(
            api_name, DeviceType.NPU, PerturbationMode.BIT_NOISE
        )
        Perturbed_value = layer.add_bit_noise(inputs)
        self.assertEqual(Perturbed_value[0], 4096.0000000000)
        self.assertEqual(Perturbed_value[1], 16777218)
        self.assertEqual(Perturbed_value[2], 1e-38)

    # 对于可迭代类型的输入, bit_noise扰动会遍历其中元素对第一个张量大于极小值的部分增加一个小值
    def test_bit_noise_layer_with_iterable_inputs(self):
        api_name = "iterable.0.forward"
        tensor_a = torch.as_tensor(4096.00048828125, dtype=torch.float32)

        inputs = [tensor_a, tensor_a, {"b": tensor_a}]
        layer = LayerFactory.create(
            api_name, DeviceType.NPU, PerturbationMode.BIT_NOISE
        )
        Perturbed_value = layer.add_bit_noise(inputs)
        self.assertEqual(Perturbed_value[0], 4096.0000000000)
        self.assertEqual(Perturbed_value[1], 4096.00048828125)
        self.assertEqual(Perturbed_value[2]["b"], 4096.00048828125)

    # 对于元素不符合要求的，bit_noise会打印告警日志
    @patch.object(logger, "warning_on_rank_0")
    def test_bit_noise_layer_with_invalid_tensor_size(self, mock_logger):
        x = torch.as_tensor([[]])
        y = torch.as_tensor([1e-9, 1e-38], dtype=torch.float32)
        layer = LayerFactory.create(
            "test_api_name", DeviceType.NPU, PerturbationMode.BIT_NOISE
        )
        layer.bit_type = torch.int16
        layer.pre_check(x)
        mock_logger.assert_called_with(
            "[msprobe] Free benchmark: For test_api_name, tensor shape must > 0."
            " Cancel adding noise."
        )
        layer.pre_check(y)
        mock_logger.assert_called_with(
            "[msprobe] Free Benchmark: For test_api_name, "
            "maximum value is less than the minimum threshold. Cancel adding noise."
        )

    # 对于输入张量，add_noise扰动因子对大于极小值的部分增加一个小值
    def test_add_noise_layer(self):
        api_name = "addnoise.0.forward"
        inputs = torch.as_tensor([1e-1, 1e-2], dtype=torch.bfloat16)
        layer = LayerFactory.create(
            api_name, DeviceType.NPU, PerturbationMode.ADD_NOISE
        )
        Perturbed_value = layer.add_noise(inputs)
        self.assertEqual(Perturbed_value[0], 1e-1 + 1e-4)
        self.assertEqual(Perturbed_value[1], 1e-2)

    # 对于可迭代类型的输入, add_noise扰动会遍历其中元素对第一个张量大于极小值的部分增加一个小值
    def test_add_noise_layer_with_iterable_inputs(self):
        api_name = "iterable.0.forward"
        tensor_a = torch.as_tensor([1e-1, 1e-2], dtype=torch.bfloat16)

        inputs = [tensor_a, tensor_a, {"b": tensor_a}]
        layer = LayerFactory.create(
            api_name, DeviceType.NPU, PerturbationMode.ADD_NOISE
        )
        Perturbed_value = layer.add_noise(inputs)
        self.assertEqual(Perturbed_value[0][0], 1e-1 + 1e-4)
        self.assertEqual(Perturbed_value[0][1], 1e-2)
        self.assertEqual(Perturbed_value[1][0], 1e-1)
        self.assertEqual(Perturbed_value[2]["b"][0], 1e-1)

    # 对于元素不符合要求的，add_noise会打印告警日志
    @patch.object(logger, "warning_on_rank_0")
    def test_add_noise_layer_with_invalid_tensor_size(self, mock_logger):
        x = torch.as_tensor([[], []], dtype=torch.float32)
        y = torch.as_tensor([1e-9, 1e-38], dtype=torch.float32)
        layer = LayerFactory.create(
            "test_api_name", DeviceType.NPU, PerturbationMode.ADD_NOISE
        )
        layer.perturbed_value = 1e-8
        layer.pre_check(x)
        mock_logger.assert_called_with(
            "[msprobe] Free benchmark: For test_api_name, tensor shape must > 0."
            " Cancel adding noise."
        )
        layer.pre_check(y)
        mock_logger.assert_called_with(
            "[msprobe] Free Benchmark: For test_api_name, "
            "maximum value is less than the minimum threshold. Cancel adding noise."
        )

    # 对于低精度输入、run cpu会升精度在cpu上计算，并会打印日志
    @patch.object(logger, "info_on_rank_0")
    def test_run_cpu_layer(self, mock_logger):
        x = torch.randn(2, 3, dtype=torch.float16)
        y = torch.randn(2, 3, dtype=torch.float16)
        out = torch.mul(x, y)
        api_name = "test_api_name"
        data_params = data_pre_deal(api_name, torch.mul, (x, y), {})
        data_params.fuzz_stage = Const.FORWARD
        data_params.original_result = out
        layer = LayerFactory.create(
            "test_api_name", DeviceType.CPU, PerturbationMode.TO_CPU
        )
        layer.handle(data_params)
        mock_logger.assert_called_with(
            "[msprobe] Free benchmark: Perturbation is to_cpu of test_api_name."
        )
        self.assertEqual(data_params.perturbed_result.dtype, torch.float32)
