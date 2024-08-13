import sys
import unittest
from unittest.mock import patch, MagicMock, Mock
import zlib

import torch
import numpy as np

from msprobe.core.data_dump.data_processor.base import ModuleBackwardInputsOutputs, ModuleForwardInputsOutputs, BaseDataProcessor
from msprobe.core.data_dump.data_processor.pytorch_processor import (
    PytorchDataProcessor,
    FreeBenchmarkDataProcessor,
    TensorDataProcessor,
    OverflowCheckDataProcessor
)

from msprobe.core.common.const import Const, OverflowConst


class TestPytorchDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = PytorchDataProcessor(self.config, self.data_writer)

    def test_get_md5_for_tensor(self):
        tensor = torch.tensor([1, 2, 3])
        expected_hash = zlib.crc32(tensor.numpy().tobytes())
        self.assertEqual(self.processor.get_md5_for_tensor(tensor), f"{expected_hash:08x}")

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

    def test_handle_tensor_extremum_nan_inf_all_nan(self):
        tensor = torch.tensor([float('nan'), float('nan')])
        result = self.processor.handle_tensor_extremum_nan_inf(tensor, 'max')
        self.assertTrue(np.isnan(result))

    def test_handle_tensor_extremum_nan_inf_all_inf(self):
        tensor = torch.tensor([float('inf'), float('inf')])
        result = self.processor.handle_tensor_extremum_nan_inf(tensor, 'max')
        self.assertTrue(np.isinf(result))

    def test_handle_tensor_extremum_nan_inf_all_negative_inf(self):
        tensor = torch.tensor([float('-inf'), float('-inf')])
        result = self.processor.handle_tensor_extremum_nan_inf(tensor, 'min')
        self.assertTrue(np.isinf(result) and result < 0)

    def test_handle_tensor_extremum_nan_inf_mixed(self):
        tensor = torch.tensor([1.0, float('nan'), 3.0, float('-inf'), 2.0])
        result_max = self.processor.handle_tensor_extremum_nan_inf(tensor, 'max')
        result_min = self.processor.handle_tensor_extremum_nan_inf(tensor, 'min')
        self.assertEqual(result_max, 3.0)
        self.assertEqual(result_min, 1.0)

    def test_handle_tensor_extremum_nan_inf_mixed_with_inf(self):
        tensor = torch.tensor([1.0, float('nan'), 3.0, float('inf'), 2.0])
        result_max = self.processor.handle_tensor_extremum_nan_inf(tensor, 'max')
        result_min = self.processor.handle_tensor_extremum_nan_inf(tensor, 'min')
        self.assertEqual(result_max, 3.0)
        self.assertEqual(result_min, 1.0)

    def test_handle_tensor_extremum_nan_inf_no_inf_nan(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result_max = self.processor.handle_tensor_extremum_nan_inf(tensor, 'max')
        result_min = self.processor.handle_tensor_extremum_nan_inf(tensor, 'min')
        self.assertEqual(result_max, 3.0)
        self.assertEqual(result_min, 1.0)

    def test_handle_tensor_extremum_nan_inf_all_inf_nan(self):
        tensor = torch.tensor([float('nan'), float('inf'), float('-inf')])
        result_max = self.processor.handle_tensor_extremum_nan_inf(tensor, 'max')
        result_min = self.processor.handle_tensor_extremum_nan_inf(tensor, 'min')
        self.assertTrue(np.isinf(result_max))
        self.assertTrue(np.isinf(result_min))

    def test_analyze_builtin(self):
        result = self.processor._analyze_builtin(slice(1, 10, 2))
        expected = {'type': 'slice', 'value': [1, 10, 2]}
        self.assertEqual(result, expected)

    def test_analyze_torch_size(self):
        size = torch.Size([3, 4, 5])
        result = self.processor._analyze_torch_size(size)
        expected = {'type': 'torch.Size', 'value': [3, 4, 5]}
        self.assertEqual(result, expected)

    def test_get_special_types(self):
        special_types = self.processor.get_special_types()
        self.assertIn(torch.Tensor, special_types)

    @patch.object(PytorchDataProcessor, 'get_md5_for_tensor')
    def test_analyze_tensor(self, get_md5_for_tensor):
        get_md5_for_tensor.return_value = 'mocked_md5'
        tensor = torch.tensor([1.0, 2.0, 3.0])
        self.config.summary_mode = 'md5'
        result = self.processor._analyze_tensor(tensor, 'suffix')
        expected = {
            'type': 'torch.Tensor',
            'dtype': str(tensor.dtype),
            'shape': tensor.shape,
            'Max': 3.0,
            'Min': 1.0,
            'Mean': 2.0,
            'Norm': torch.norm(tensor).item(),
            'requires_grad': tensor.requires_grad,
            'md5': 'mocked_md5'
        }
        self.assertDictEqual(expected, result)

    def test_analyze_tensor_with_empty_tensor(self):
        tensor = torch.tensor([])
        result = self.processor._analyze_tensor(tensor, 'suffix')
        self.assertEqual(result['Max'], None)
        self.assertEqual(result['Min'], None)
        self.assertEqual(result['Mean'], None)
        self.assertEqual(result['Norm'], None)

    def test_analyze_tensor_with_inf_and_nan(self):
        tensor = torch.tensor([1.0, float('inf'), float('nan'), -float('inf')])
        result = self.processor._analyze_tensor(tensor, 'suffix')
        self.assertEqual(result['Max_except_inf_nan'], 1.0)
        self.assertEqual(result['Min_except_inf_nan'], 1.0)


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
        tensor = torch.tensor([1.0, 2.0, 3.0])
        suffix = 'suffix'
        result = self.processor._analyze_tensor(tensor, suffix)
        mock_save.assert_called_once()
        expected = {
            'type': 'torch.Tensor',
            'dtype': 'torch.float32',
            'shape': tensor.shape,
            'Max': 3.0,
            'Min': 1.0,
            'Mean': 2.0,
            'Norm': torch.norm(tensor).item(),
            'requires_grad': False,
            'data_name': 'test_api.input.suffix.pt'
        }
        self.assertEqual(expected, result)


class TestOverflowCheckDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.config.overflow_nums = 1
        self.data_writer = MagicMock()
        self.processor = OverflowCheckDataProcessor(self.config, self.data_writer)
        self.current_api_or_module_name = "test_api_name"
        self.api_data_category = "input"
        sys.modules['torch_npu'] = Mock()
        sys.modules['torch_npu.npu'] = Mock()
        sys.modules['torch_npu._C'] = Mock()

    @patch('torch.save')
    def test_maybe_save_overflow_data_and_check_overflow_times(self, mock_save):
        self.processor.has_overflow = True
        self.processor.real_overflow_nums = 0
        self.processor.cached_tensors_and_file_paths = {'dummy_path': torch.tensor([1.0, 2.0, 3.0])}
        self.processor.maybe_save_overflow_data_and_check_overflow_times()
        mock_save.assert_called_once()
        self.processor.maybe_save_overflow_data_and_check_overflow_times()
        self.assertEqual(self.processor.real_overflow_nums, 2)

    @patch('os.getenv', return_value=Const.ENV_ENABLE)
    def test_overflow_debug_mode_enable(self, mock_getenv):
        result = self.processor.overflow_debug_mode_enable()
        self.assertTrue(result)
        mock_getenv.assert_called_once_with(OverflowConst.OVERFLOW_DEBUG_MODE_ENABLE, Const.ENV_DISABLE)

    @patch('msprobe.core.data_dump.data_processor.pytorch_processor.is_gpu', return_value=True)
    def test_analyze_maybe_overflow_tensor(self, _):
        tensor_json = {'Max': float('inf'), 'Min': 1.0}
        self.processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertTrue(self.processor.has_overflow)

    @patch('msprobe.core.common.file_check.path_len_exceeds_limit', return_value=False)
    @patch.object(BaseDataProcessor, 'get_save_file_path', return_value=['test_api_name', 'test_api_name.0.forward.input.pt'])
    def test_analyze_tensor(self, mock_path_len_exceeds_limit, _):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        suffix = 'suffix'
        expected = {'Max': 3.0, 'Min': 1.0, 'data_name': 'test_api_name'}
        with patch.object(PytorchDataProcessor, '_analyze_tensor', return_value={'Max': 3.0, 'Min': 1.0}) as mock_super_analyze_tensor:
            result = self.processor._analyze_tensor(tensor, suffix)
            mock_super_analyze_tensor.assert_called_once_with(tensor, suffix)
            mock_path_len_exceeds_limit.assert_called_once()
            self.assertEqual(expected, result)

    def test_analyze_backward(self):
        def func(_):
            self.processor.has_overflow = True
        with patch.object(BaseDataProcessor, "analyze_backward", return_value={"min", 0}):
            with patch.object(OverflowCheckDataProcessor, "maybe_save_overflow_data_and_check_overflow_times"):
                api_info = self.processor.analyze_backward("name", "module", "module_input_output")
            self.assertFalse(self.processor.has_overflow)
            self.assertIsNone(api_info)
            with patch.object(OverflowCheckDataProcessor, "maybe_save_overflow_data_and_check_overflow_times", new=func):
                api_info = self.processor.analyze_backward("name", "module", "module_input_output")
            self.assertTrue(self.processor.has_overflow)
            self.assertEqual(api_info, {"min", 0})

    def test_analyze_forward(self):
        def func(_):
            self.processor.has_overflow = True
        with patch.object(BaseDataProcessor, "analyze_forward", return_value={"min", 0}):
            with patch.object(OverflowCheckDataProcessor, "maybe_save_overflow_data_and_check_overflow_times"):
                api_info = self.processor.analyze_forward("name", "module", "module_input_output")
            self.assertFalse(self.processor.has_overflow)
            self.assertIsNone(api_info)
            with patch.object(OverflowCheckDataProcessor, "maybe_save_overflow_data_and_check_overflow_times", new=func):
                api_info = self.processor.analyze_forward("name", "module", "module_input_output")
            self.assertTrue(self.processor.has_overflow)
            self.assertEqual(api_info, {"min", 0})


class TestFreeBenchmarkDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = FreeBenchmarkDataProcessor(self.config, self.data_writer)

    @patch('msprobe.pytorch.free_benchmark.FreeBenchmarkCheck.pre_forward')
    def test_analyze_pre_forward(self, mock_pre_forward):
        module_io = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3}, output=None)
        self.processor.analyze_pre_forward('test_pre_forward', None, module_io)
        mock_pre_forward.assert_called_once()

    @patch('msprobe.pytorch.free_benchmark.FreeBenchmarkCheck.forward', return_value=(None, []))
    def test_analyze_forward(self, mock_forward):
        module_io = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3}, output=(4, 5))
        self.processor.analyze_forward('test_forward', None, module_io)
        mock_forward.assert_called_once()

    @patch('msprobe.pytorch.free_benchmark.FreeBenchmarkCheck.backward')
    def test_analyze_backward(self, mock_backward):
        module_io = ModuleBackwardInputsOutputs(grad_output=(torch.tensor([1.0, 2.0]),), grad_input=None)
        self.processor.analyze_backward('test_backward', None, module_io)
        mock_backward.assert_called_once()
