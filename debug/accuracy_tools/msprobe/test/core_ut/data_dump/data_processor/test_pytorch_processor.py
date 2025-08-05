import hashlib
import os
import sys
import unittest
import zlib
from unittest.mock import patch, MagicMock, Mock

import numpy as np
import torch
from msprobe.core.common.log import logger
from msprobe.core.data_dump.data_processor.base import ModuleBackwardInputsOutputs, ModuleForwardInputsOutputs, \
    BaseDataProcessor
from msprobe.core.data_dump.data_processor.pytorch_processor import (
    PytorchDataProcessor,
    FreeBenchmarkDataProcessor,
    TensorDataProcessor,
    OverflowCheckDataProcessor,
    TensorStatInfo,
    KernelDumpDataProcessor
)
from torch import distributed as dist
from torch._subclasses import FakeTensorMode


class TestPytorchDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = PytorchDataProcessor(self.config, self.data_writer)

    def test_get_md5_for_tensor(self):
        tensor = torch.tensor([1, 2, 3])
        expected_hash = zlib.crc32(tensor.numpy().tobytes())
        self.assertEqual(self.processor.get_md5_for_tensor(tensor), f"{expected_hash:08x}")

    def test_get_md5_for_tensor_bfloat16(self):
        tensor_bfloat16 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        expected_hash = zlib.crc32(tensor_bfloat16.float().cpu().detach().numpy().tobytes())
        result_hash = self.processor.get_md5_for_tensor(tensor_bfloat16)
        self.assertEqual(result_hash, f"{expected_hash:08x}")

    def test_analyze_device_in_kwargs(self):
        device = torch.device('cuda:0')
        result = self.processor.analyze_device_in_kwargs(device)
        expected = {'type': 'torch.device', 'value': 'cuda:0'}
        self.assertEqual(result, expected)

    def test_analyze_dtype_in_kwargs(self):
        dtype = torch.float32
        result = self.processor.analyze_dtype_in_kwargs(dtype)
        expected = {'type': 'torch.dtype', 'value': 'torch.float32'}
        self.assertEqual(result, expected)

    @staticmethod
    def mock_tensor(is_meta):
        tensor = MagicMock()
        tensor.is_meta = is_meta
        return tensor

    def test_get_stat_info_with_meta_tensor(self):
        mock_data = self.mock_tensor(is_meta=True)
        result = self.processor.get_stat_info(mock_data)
        self.assertIsInstance(result, TensorStatInfo)

    def test_get_stat_info_with_fake_tensor(self):
        with FakeTensorMode() as fake_tensor_mode:
            fake_tensor = fake_tensor_mode.from_tensor(torch.randn(1, 2, 3))
        result = self.processor.get_stat_info(fake_tensor)
        self.assertIsNone(result.max)
        self.assertIsNone(result.min)
        self.assertIsNone(result.mean)
        self.assertIsNone(result.norm)

    def test_get_stat_info_float(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, 3.0)
        self.assertEqual(result.min, 1.0)
        self.assertEqual(result.mean, 2.0)
        self.assertEqual(result.norm, torch.norm(tensor).item())

    def test_get_stat_info_int(self):
        tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = self.processor.get_stat_info(tensor)

        self.assertEqual(result.max, 3)
        self.assertEqual(result.min, 1)
        self.assertEqual(result.mean, 2)
        self.assertEqual(result.norm, torch.norm(tensor.float()).item())

    def test_get_stat_info_empty(self):
        tensor = torch.tensor([])
        result = self.processor.get_stat_info(tensor)
        self.assertIsNone(result.max)
        self.assertIsNone(result.min)
        self.assertIsNone(result.mean)
        self.assertIsNone(result.norm)

    def test_get_stat_info_bool(self):
        tensor = torch.tensor([True, False, True])
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, True)
        self.assertEqual(result.min, False)
        self.assertIsNone(result.mean)
        self.assertIsNone(result.norm)

    def test_get_stat_info_with_scalar_tensor(self):
        scalar_tensor = torch.tensor(42.0)
        result = self.processor.get_stat_info(scalar_tensor)
        self.assertIsInstance(result, TensorStatInfo)
        self.assertEqual(result.max, 42.0)
        self.assertEqual(result.min, 42.0)
        self.assertEqual(result.mean, 42.0)
        self.assertEqual(result.norm, 42.0)

    def test_get_stat_info_with_complex_tensor(self):
        complex_tensor = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
        result = self.processor.get_stat_info(complex_tensor)
        expected_max = np.abs(np.array([1 + 2j, 3 + 4j])).max().item()
        expected_min = np.abs(np.array([1 + 2j, 3 + 4j])).min().item()
        expected_mean = np.abs(np.array([1 + 2j, 3 + 4j])).mean().item()
        self.assertIsInstance(result, TensorStatInfo)
        self.assertAlmostEqual(result.max, expected_max, places=6)
        self.assertAlmostEqual(result.min, expected_min, places=6)
        self.assertAlmostEqual(result.mean, expected_mean, places=6)

    def test_analyze_builtin(self):
        result = self.processor._analyze_builtin(slice(1, torch.tensor(10, dtype=torch.int32), np.int64(2)))
        expected = {'type': 'slice', 'value': [1, 10, 2]}
        self.assertEqual(result, expected)

        result = self.processor._analyze_builtin(slice(torch.tensor([1, 2], dtype=torch.int32), None, None))
        expected = {'type': 'slice', 'value': [None, None, None]}
        self.assertEqual(result, expected)

    def test_process_group_hash(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(backend='gloo', world_size=1, rank=0)
        process_group_element = dist.group.WORLD
        result = self.processor.process_group_hash(process_group_element)
        expected = f"{zlib.crc32(str([0]).encode('utf-8')):08x}"
        self.assertEqual(result, expected)

    def test_analyze_torch_size(self):
        size = torch.Size([3, 4, 5])
        result = self.processor._analyze_torch_size(size)
        expected = {'type': 'torch.Size', 'value': [3, 4, 5]}
        self.assertEqual(result, expected)

    def test_analyze_memory_format(self):
        memory_format_element = torch.contiguous_format
        result = self.processor._analyze_memory_format(memory_format_element)
        expected = {'type': 'torch.memory_format', 'format': 'contiguous_format'}
        self.assertEqual(result, expected)

    def test_analyze_process_group(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(backend='gloo', world_size=1, rank=0)
        process_group_element = dist.group.WORLD
        result = self.processor._analyze_process_group(process_group_element)
        expected = {
            'type': 'torch.ProcessGroup',
            'group_ranks': [0],
            'group_id': f"{zlib.crc32(str([0]).encode('utf-8')):08x}"
        }
        self.assertEqual(result, expected)

    def test_analyze_reduce_op_successful(self):
        arg = dist.ReduceOp.SUM
        result = self.processor._analyze_reduce_op(arg)
        expected = {'type': 'torch.distributed.ReduceOp', 'value': 'RedOpType.SUM'}
        self.assertEqual(result, expected)

    @patch.object(logger, 'warning')
    def test_analyze_reduce_op_failed(self, mock_logger_warning):
        class TestReduceOp:
            def __str__(self):
                raise Exception("failed to convert str type")

        arg = TestReduceOp()
        self.processor._analyze_reduce_op(arg)
        mock_logger_warning.assert_called_with(
            "Failed to get value of torch.distributed.ReduceOp with error info: failed to convert str type."
        )

    def test_get_special_types(self):
        special_types = self.processor.get_special_types()
        self.assertIn(torch.Tensor, special_types)

    def test_analyze_single_element_torch_size(self):
        size_element = torch.Size([2, 3])
        result = self.processor.analyze_single_element(size_element, [])
        self.assertEqual(result, self.processor._analyze_torch_size(size_element))

    def test_analyze_single_element_memory_size(self):
        memory_format_element = torch.contiguous_format
        result = self.processor.analyze_single_element(memory_format_element, [])
        self.assertEqual(result, self.processor._analyze_memory_format(memory_format_element))

    def test_analyze_single_element_process_group(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(backend='gloo', world_size=1, rank=0)
        process_group_element = dist.group.WORLD
        result = self.processor.analyze_single_element(process_group_element, [])
        self.assertEqual(result, self.processor._analyze_process_group(process_group_element))

    def test_analyze_single_element_numpy_conversion(self):
        numpy_element = np.int32(5)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": 'int32', "value": 5}
        self.assertEqual(result, expected)

        numpy_element = np.float32(3.14)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": 'float32', "value": 3.140000104904175}
        self.assertEqual(result, expected)

        numpy_element = np.bool_(True)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": 'bool_', "value": True}
        self.assertEqual(result, expected)

        numpy_element = np.str_("abc")
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": 'str_', "value": "abc"}
        self.assertEqual(result, expected)

        numpy_element = np.byte(1)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": 'int8', "value": 1}
        self.assertEqual(result, expected)

        numpy_element = np.complex128(1 + 2j)
        result = self.processor.analyze_single_element(numpy_element, [])
        expected = {"type": 'complex128', "value": (1 + 2j)}
        self.assertEqual(result, expected)

    def test_analyze_single_element_tensor(self):
        tensor_element = torch.tensor([1, 2, 3])
        result = self.processor.analyze_single_element(tensor_element, ['tensor'])
        expected_result = self.processor._analyze_tensor(tensor_element, "tensor")
        self.assertEqual(result, expected_result, f"{result} {expected_result}")

    def test_analyze_single_element_bool(self):
        bool_element = True
        result = self.processor.analyze_single_element(bool_element, [])
        expected_result = self.processor._analyze_builtin(bool_element)
        self.assertEqual(result, expected_result)

    def test_analyze_single_element_builtin_ellipsis(self):
        result = self.processor.analyze_single_element(Ellipsis, [])
        expected_result = self.processor._analyze_builtin(Ellipsis)
        self.assertEqual(result, expected_result)

    @patch.object(PytorchDataProcessor, 'get_md5_for_tensor')
    def test_analyze_tensor(self, get_md5_for_tensor):
        get_md5_for_tensor.return_value = 'mocked_md5'
        tensor = torch.tensor([1.0, 2.0, 3.0])
        self.config.summary_mode = 'md5'
        self.config.async_dump = False
        result = self.processor._analyze_tensor(tensor, 'suffix')
        expected = {
            'type': 'torch.Tensor',
            'dtype': str(tensor.dtype),
            'shape': tensor.shape,
            'requires_grad': tensor.requires_grad
        }
        result.pop('tensor_stat_index', None)
        result.pop('md5_index', None)
        self.assertDictEqual(expected, result)

    def test_analyze_tensor_with_empty_tensor(self):
        tensor = torch.tensor([])
        result = self.processor._analyze_tensor(tensor, 'suffix')

        self.assertEqual(result['type'], "torch.Tensor")
        self.assertEqual(result['dtype'], 'torch.float32')
        self.assertEqual(result['shape'], torch.Size([0]))
        self.assertEqual(result['requires_grad'], False)


class TestTensorDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = TensorDataProcessor(self.config, self.data_writer)
        self.data_writer.dump_tensor_data_dir = "./dump_data"
        self.processor.current_api_or_module_name = "test_api"
        self.processor.api_data_category = "input"

    @patch('torch.save')
    def test_analyze_tensor(self, mock_save):
        self.config.framework = "pytorch"
        self.config.async_dump = False
        tensor = torch.tensor([1.0, 2.0, 3.0])
        suffix = 'suffix'
        result = self.processor._analyze_tensor(tensor, suffix)
        mock_save.assert_called_once()
        expected = {
            'type': 'torch.Tensor',
            'dtype': 'torch.float32',
            'shape': tensor.shape,
            'requires_grad': False,
            'data_name': 'test_api.input.suffix.pt'
        }
        result.pop('tensor_stat_index', None)
        self.assertEqual(expected, result)


class TestOverflowCheckDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.config.overflow_nums = 1
        self.data_writer = MagicMock()
        self.processor = OverflowCheckDataProcessor(self.config, self.data_writer)
        self.processor.support_inf_nan = True
        self.processor.has_overflow = False
        self.current_api_or_module_name = "test_api_name"
        self.api_data_category = "input"
        sys.modules['torch_npu'] = Mock()
        sys.modules['torch_npu.npu'] = Mock()
        sys.modules['torch_npu.npu.utils'] = Mock()
        self.tensor_json = {
            'tensor_stat_index': 123  # 默认情况下 tensor_stat_index 存在
        }

    def test_is_terminated(self):
        self.processor.overflow_nums = -1
        self.assertFalse(self.processor.is_terminated)

        self.processor.real_overflow_nums = 2
        self.processor.overflow_nums = 2
        self.assertTrue(self.processor.is_terminated)

        self.processor.real_overflow_nums = 1
        self.assertFalse(self.processor.is_terminated)

    def test_analyze_forward_input(self):
        with patch.object(BaseDataProcessor, "analyze_forward_input", return_value={"name": 1}):
            api_info = self.processor.analyze_forward_input("name", "module", "module_input_output")
            self.assertEqual(self.processor.cached_api_info, {"name": 1})
            self.assertIsNone(api_info)

    def test_analyze_forward_output(self):
        def func(_):
            self.processor.has_overflow = True

        with patch.object(BaseDataProcessor, "analyze_forward_output", return_value={"name": {"output": 2}}), \
                patch.object(OverflowCheckDataProcessor, "handle_overflow", new=func):
            self.processor.cached_api_info = {"name": {"intput": 1}}
            api_info = self.processor.analyze_forward_output("name", "module", "module_input_output")
            self.assertEqual(api_info, {"name": {"intput": 1, "output": 2}})

    def test_analyze_forward(self):
        def func(_):
            self.processor.has_overflow = True

        with patch.object(BaseDataProcessor, "analyze_forward", return_value={"min", 0}):
            with patch.object(OverflowCheckDataProcessor, "handle_overflow"):
                api_info = self.processor.analyze_forward("name", "module", "module_input_output")
            self.assertFalse(self.processor.has_overflow)
            self.assertIsNone(api_info)

            with patch.object(OverflowCheckDataProcessor, "handle_overflow", new=func):
                api_info = self.processor.analyze_forward("name", "module", "module_input_output")
            self.assertTrue(self.processor.has_overflow)
            self.assertEqual(api_info, {"min", 0})

    def test_analyze_backward(self):
        def func(_):
            self.processor.has_overflow = True

        with patch.object(BaseDataProcessor, "analyze_backward", return_value={"min", 0}):
            with patch.object(OverflowCheckDataProcessor, "handle_overflow"):
                api_info = self.processor.analyze_backward("name", "module", "module_input_output")
            self.assertFalse(self.processor.has_overflow)
            self.assertIsNone(api_info)

            with patch.object(OverflowCheckDataProcessor, "handle_overflow", new=func):
                api_info = self.processor.analyze_backward("name", "module", "module_input_output")
            self.assertTrue(self.processor.has_overflow)
            self.assertEqual(api_info, {"min", 0})

    @patch('msprobe.core.data_dump.data_processor.pytorch_processor.save_pt')
    def test_handle_overflow(self, mock_save):
        self.processor.has_overflow = True
        self.processor.real_overflow_nums = 0
        self.processor.cached_tensors_and_file_paths = {'dump_path': torch.tensor([1.0, 2.0, 3.0])}
        self.processor.handle_overflow()
        mock_save.assert_called_once()
        self.processor.has_overflow = True
        self.processor.handle_overflow()
        self.assertEqual(self.processor.real_overflow_nums, 2)

    @patch('msprobe.core.data_dump.data_processor.pytorch_processor.is_gpu')
    def test_is_support_inf_nan_gpu(self, mock_gpu_device):
        self.processor.support_inf_nan = None
        mock_gpu_device.return_value = True
        self.processor._is_support_inf_nan()
        self.assertTrue(self.processor.support_inf_nan)

    def test_max_tensor_or_min_tensor_is_none(self):
        # 让 get_buffer_values_max 和 get_buffer_values_min 返回 None
        self.processor.data_writer.get_buffer_values_max.return_value = None
        self.processor.data_writer.get_buffer_values_min.return_value = None

        # 在该情况下应该直接返回，不做任何改变
        self.processor._analyze_maybe_overflow_tensor(self.tensor_json)

        # 确保 has_overflow 没有被设置
        self.assertFalse(self.processor.has_overflow)

    def test_tensor_is_inf_or_nan(self):
        # 模拟 max_tensor 为 Inf
        self.processor.data_writer.get_buffer_values_max.return_value = torch.tensor(float('inf'))
        self.processor.data_writer.get_buffer_values_min.return_value = torch.tensor(1.0)

        # 测试应该设置 has_overflow 为 True
        self.processor._analyze_maybe_overflow_tensor(self.tensor_json)
        self.assertTrue(self.processor.has_overflow)

        # 模拟 min_tensor 为 NaN
        self.processor.data_writer.get_buffer_values_max.return_value = torch.tensor(1.0)
        self.processor.data_writer.get_buffer_values_min.return_value = torch.tensor(float('nan'))

        # 测试应该设置 has_overflow 为 True
        self.processor._analyze_maybe_overflow_tensor(self.tensor_json)
        self.assertTrue(self.processor.has_overflow)

    def test_normal_tensor(self):
        # 模拟正常的 max_tensor 和 min_tensor
        self.processor.data_writer.get_buffer_values_max.return_value = torch.tensor(1.0)
        self.processor.data_writer.get_buffer_values_min.return_value = torch.tensor(-1.0)

        # 在正常情况下不应该改变 has_overflow
        self.processor._analyze_maybe_overflow_tensor(self.tensor_json)
        self.assertFalse(self.processor.has_overflow)

    @patch('msprobe.core.common.file_utils.path_len_exceeds_limit', return_value=False)
    @patch.object(BaseDataProcessor, 'get_save_file_path',
                  return_value=['test_api_name', 'test_api_name.0.forward.input.pt'])
    def test_analyze_tensor(self, mock_path_len_exceeds_limit, _):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        suffix = 'suffix'
        expected = {'Max': 3.0, 'Min': 1.0, 'data_name': 'test_api_name'}
        with patch.object(PytorchDataProcessor, '_analyze_tensor',
                          return_value={'Max': 3.0, 'Min': 1.0}) as mock_super_analyze_tensor:
            result = self.processor._analyze_tensor(tensor, suffix)
            mock_super_analyze_tensor.assert_called_once_with(tensor, suffix)
            mock_path_len_exceeds_limit.assert_called_once()
            self.assertEqual(expected, result)


class TestFreeBenchmarkDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = FreeBenchmarkDataProcessor(self.config, self.data_writer)

    def test_update_iter(self):
        self.processor.checker = MagicMock()
        current_iter = 5
        self.processor.update_iter(current_iter)
        self.processor.checker.update_iter.assert_called_once_with(current_iter)

    def test_update_unequal_rows_empty(self):
        self.processor.update_unequal_rows([])
        self.data_writer.write_data_to_csv.assert_not_called()

    @patch('msprobe.pytorch.free_benchmark.FreeBenchmarkCheck.pre_forward')
    def test_analyze_forward_input(self, mock_pre_forward):
        module_io = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3}, output=None)
        self.processor.analyze_forward_input('test_analyze_forward_input', None, module_io)
        mock_pre_forward.assert_called_once()

    @patch('msprobe.pytorch.free_benchmark.FreeBenchmarkCheck.forward', return_value=(None, []))
    def test_analyze_forward_output(self, mock_forward):
        module_io = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3}, output=(4, 5))
        self.processor.analyze_forward_output('test_analyze_forward_output', None, module_io)
        mock_forward.assert_called_once()

    def test_analyze_forward_output_if_fix_branch(self):
        self.processor.checker = MagicMock()
        name = "test_module"
        module = MagicMock()
        module_input_output = MagicMock()
        module_input_output.args = (1, 2)
        module_input_output.kwargs = {'key': 'value'}
        module_input_output.output = "some_output"

        new_output = "new_output_value"
        unequal_rows = []
        self.processor.checker.forward.return_value = (new_output, unequal_rows)
        self.processor.checker.if_fix.return_value = True
        self.processor.analyze_forward_output(name, module, module_input_output)
        self.processor.checker.if_fix.assert_called_once()
        self.assertTrue(self.processor._return_forward_new_output)
        self.assertEqual(self.processor._forward_new_output, new_output)

    @patch('msprobe.pytorch.free_benchmark.FreeBenchmarkCheck.backward')
    def test_analyze_backward(self, mock_backward):
        module_io = ModuleBackwardInputsOutputs(grad_output=(torch.tensor([1.0, 2.0]),), grad_input=None)
        self.processor.analyze_backward('test_backward', None, module_io)
        mock_backward.assert_called_once()


class TestKernelDumpDataProcessor(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = KernelDumpDataProcessor(self.config, self.data_writer)

    @patch.object(logger, 'warning')
    def test_print_unsupported_log(self, mock_logger_warning):
        self.processor._print_unsupported_log("test_api_name")
        mock_logger_warning.assert_called_with("The kernel dump does not support the test_api_name API.")

    @patch('msprobe.core.data_dump.data_processor.pytorch_processor.is_gpu')
    @patch.object(logger, 'warning')
    def test_analyze_pre_forward_with_gpu(self, mock_logger_warning, mock_is_gpu):
        mock_is_gpu = True
        self.processor.analyze_forward_input("test_api_name", None, None)
        mock_logger_warning.assert_called_with(
            "The current environment is not a complete NPU environment, and kernel dump cannot be used.")
        self.assertFalse(self.processor.enable_kernel_dump)

    @patch('msprobe.core.data_dump.data_processor.pytorch_processor.is_gpu', new=False)
    @patch('msprobe.core.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.analyze_element')
    @patch.object(logger, 'warning')
    def test_analyze_pre_forward_with_not_gpu(self, mock_logger_warning, mock_analyze_element):
        self.config.is_backward_kernel_dump = True
        mock_module = MagicMock()
        mock_module_input_output = MagicMock()
        self.processor.analyze_forward_input("test_api_name", mock_module, mock_module_input_output)
        mock_module.forward.assert_called_once()
        mock_analyze_element.assert_called()
        mock_logger_warning.assert_called_with("The kernel dump does not support the test_api_name API.")
        self.assertFalse(self.processor.enable_kernel_dump)

    @patch('msprobe.core.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.stop_kernel_dump')
    @patch.object(logger, 'info')
    def test_analyze_forward_successfully(self, mock_logger_info, mock_stop_kernel_dump):
        self.processor.enable_kernel_dump = True
        self.processor.config.is_backward_kernel_dump = False
        self.processor.analyze_forward_output('test_api_name', None, None)
        self.assertFalse(self.processor.enable_kernel_dump)
        mock_stop_kernel_dump.assert_called_once()
        mock_logger_info.assert_called_with("The kernel data of test_api_name is dumped successfully.")

    @patch('msprobe.core.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.analyze_element')
    @patch.object(logger, 'warning')
    def test_analyze_backward_unsuccessfully(self, mock_logger_warning, mock_analyze_element):
        self.processor.enable_kernel_dump = True
        self.processor.is_found_grad_input_tensor = False
        mock_module_input_output = MagicMock()
        self.processor.analyze_backward("test_api_name", None, mock_module_input_output)
        mock_analyze_element.assert_called_once()
        mock_logger_warning.assert_called_with("The kernel dump does not support the test_api_name API.")
        self.assertFalse(self.processor.enable_kernel_dump)

    @patch('msprobe.core.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.stop_kernel_dump')
    @patch('msprobe.core.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.start_kernel_dump')
    @patch('msprobe.core.data_dump.data_processor.pytorch_processor.KernelDumpDataProcessor.analyze_element')
    @patch.object(logger, 'info')
    def test_analyze_backward_successfully(self, mock_logger_info, mock_analyze_element, mock_start, mock_stop):
        self.processor.enable_kernel_dump = True
        self.processor.is_found_grad_input_tensor = True
        self.processor.forward_output_tensor = MagicMock()
        mock_module_input_output = MagicMock()
        self.processor.analyze_backward("test_api_name", None, mock_module_input_output)
        mock_analyze_element.assert_called_once()
        self.assertFalse(self.processor.enable_kernel_dump)
        self.processor.forward_output_tensor.backward.assert_called_once()
        mock_start.assert_called_once()
        mock_stop.assert_called_once()
        mock_logger_info.assert_called_with("The kernel data of test_api_name is dumped successfully.")

    def test_clone_tensor(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        clone_tensor = self.processor.clone_and_detach_tensor(tensor)
        self.assertTrue(torch.equal(tensor, clone_tensor))
        self.assertFalse(clone_tensor.requires_grad)

        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        clone_tensor = self.processor.clone_and_detach_tensor(tensor)
        self.assertTrue(torch.equal(tensor, clone_tensor))
        self.assertTrue(clone_tensor.requires_grad)

        tensor1 = torch.tensor([1.0], requires_grad=True)
        tensor2 = torch.tensor([1.0])
        input_tuple = (tensor1, tensor2)
        clone_tuple = self.processor.clone_and_detach_tensor(input_tuple)
        self.assertEqual(len(input_tuple), len(clone_tuple))
        self.assertTrue(clone_tuple[0].requires_grad)
        self.assertFalse(clone_tuple[1].requires_grad)

        input_list = [tensor1, tensor2]
        clone_list = self.processor.clone_and_detach_tensor(input_list)
        self.assertEqual(len(input_list), len(clone_list))
        self.assertTrue(clone_tuple[0].requires_grad)
        self.assertFalse(clone_tuple[1].requires_grad)

        input_dict = {'tensor1': tensor1, 'tensor2': tensor2}
        clone_dict = self.processor.clone_and_detach_tensor(input_dict)
        self.assertEqual(len(clone_dict), len(input_dict))
        self.assertTrue(clone_dict["tensor1"].requires_grad)
        self.assertFalse(clone_dict["tensor2"].requires_grad)

        non_tensor_input = 1
        result = self.processor.clone_and_detach_tensor(non_tensor_input)
        self.assertEqual(result, non_tensor_input)

    def test_analyze_single_element_with_output_grad(self):
        self.processor.is_found_output_tensor = False
        tensor = torch.tensor([1.0], requires_grad=True)
        self.processor.analyze_single_element(tensor, None)
        self.assertTrue(self.processor.is_found_output_tensor)

    def test_analyze_single_element_without_output_grad(self):
        self.processor.is_found_output_tensor = False
        tensor = torch.tensor([1.0])
        self.processor.analyze_single_element(tensor, None)
        self.assertFalse(self.processor.is_found_output_tensor)

    def test_analyze_single_element_with_grad_input(self):
        self.processor.is_found_output_tensor = True
        self.processor.is_found_grad_input_tensor = False
        tensor = torch.tensor([1.0])
        self.processor.analyze_single_element(tensor, None)
        self.assertTrue(self.processor.is_found_grad_input_tensor)

    def test_analyze_single_element_without_grad_input(self):
        self.processor.is_found_output_tensor = True
        self.processor.is_found_grad_input_tensor = True
        tensor = torch.tensor([1.0])
        self.processor.analyze_single_element(tensor, None)
        self.assertTrue(self.processor.is_found_grad_input_tensor)

    def test_reset_status(self):
        self.processor.enable_kernel_dump = False
        self.processor.is_found_output_tensor = True
        self.processor.is_found_grad_input_tensor = True
        self.processor.forward_args = 0
        self.processor.forward_kwargs = 1
        self.processor.forward_output_tensor = 2
        self.processor.grad_input_tensor = 3

        self.processor.reset_status()

        self.assertTrue(self.processor.enable_kernel_dump)
        self.assertFalse(self.processor.is_found_output_tensor)
        self.assertFalse(self.processor.is_found_grad_input_tensor)
        self.assertIsNone(self.processor.forward_args)
        self.assertIsNone(self.processor.forward_kwargs)
        self.assertIsNone(self.processor.forward_output_tensor)
        self.assertIsNone(self.processor.grad_input_tensor)
