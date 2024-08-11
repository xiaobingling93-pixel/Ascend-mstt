import sys
import unittest
from unittest.mock import patch, MagicMock, Mock
import zlib

# mock_torch_npu = MagicMock()
# modules = {
#     'torch_npu': mock_torch_npu,
#     'torch_npu.npu': mock_torch_npu.npu,
# }
# patcher = patch.dict('sys.modules', modules)
# patcher.start()

import torch
from msprobe.core.data_dump.data_processor.base import ModuleBackwardInputsOutputs, ModuleForwardInputsOutputs, BaseDataProcessor
from msprobe.core.data_dump.data_processor.pytorch_processor import (
    PytorchDataProcessor,
    FreeBenchmarkDataProcessor,
    TensorDataProcessor,
    KernelDumpDataProcessor,
    OverflowCheckDataProcessor
)

from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.file_check import change_mode, path_len_exceeds_limit
from msprobe.core.common.const import Const, OverflowConst, FileCheckConst
sys.modules['torch_npu'] = Mock()
sys.modules['torch_npu.npu'] = Mock()
sys.modules['torch_npu._C'] = Mock()
torch_npu = sys.modules['torch_npu']



class TestPytorchDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = PytorchDataProcessor(self.config, self.data_writer)

    def test_get_md5_for_tensor(self):
        tensor = torch.tensor([1, 2, 3])
        expected_md5 = zlib.crc32(tensor.numpy().tobytes())
        self.assertEqual(self.processor.get_md5_for_tensor(tensor), f"{expected_md5:08x}")

    def test_analyze_device_in_kwargs(self):
        device = torch.device('npu:0')
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
        result = PytorchDataProcessor.handle_tensor_extremum_nan_inf(tensor, 'max')
        self.assertTrue(torch.isnan(result))

    def test_handle_tensor_extremum_nan_inf_all_inf(self):
        tensor = torch.tensor([float('inf'), float('inf')])
        result = PytorchDataProcessor.handle_tensor_extremum_nan_inf(tensor, 'max')
        self.assertTrue(torch.isinf(result))

    def test_handle_tensor_extremum_nan_inf_all_negative_inf(self):
        tensor = torch.tensor([float('-inf'), float('-inf')])
        result = PytorchDataProcessor.handle_tensor_extremum_nan_inf(tensor, 'min')
        self.assertTrue(torch.isinf(result) and result < 0)

    def test_handle_tensor_extremum_nan_inf_mixed(self):
        tensor = torch.tensor([1.0, float('nan'), 3.0, float('-inf'), 2.0])
        result_max = PytorchDataProcessor.handle_tensor_extremum_nan_inf(tensor, 'max')
        result_min = PytorchDataProcessor.handle_tensor_extremum_nan_inf(tensor, 'min')
        self.assertEqual(result_max, 3.0)
        self.assertEqual(result_min, 1.0)

    def test_handle_tensor_extremum_nan_inf_mixed_with_inf(self):
        tensor = torch.tensor([1.0, float('nan'), 3.0, float('inf'), 2.0])
        result_max = PytorchDataProcessor.handle_tensor_extremum_nan_inf(tensor, 'max')
        result_min = PytorchDataProcessor.handle_tensor_extremum_nan_inf(tensor, 'min')
        self.assertEqual(result_max, 3.0)
        self.assertEqual(result_min, 1.0)

    def test_handle_tensor_extremum_nan_inf_no_inf_nan(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result_max = PytorchDataProcessor.handle_tensor_extremum_nan_inf(tensor, 'max')
        result_min = PytorchDataProcessor.handle_tensor_extremum_nan_inf(tensor, 'min')
        self.assertEqual(result_max, 3.0)
        self.assertEqual(result_min, 1.0)

    def test_handle_tensor_extremum_nan_inf_all_inf_nan(self):
        tensor = torch.tensor([float('nan'), float('inf'), float('-inf')])
        result_max = PytorchDataProcessor.handle_tensor_extremum_nan_inf(tensor, 'max')
        result_min = PytorchDataProcessor.handle_tensor_extremum_nan_inf(tensor, 'min')
        self.assertTrue(torch.isnan(result_max))
        self.assertTrue(torch.isnan(result_min))

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

    @patch('torch.save')
    @patch('msprobe.core.common.file_check.path_len_exceeds_limit', return_value=False)
    @patch('msprobe.core.common.file_check.change_mode')
    def test_analyze_tensor(self, mock_change_mode, mock_save):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        suffix = 'suffix'
        result = self.processor._analyze_tensor(tensor, suffix)
        mock_save.assert_called_once()
        mock_change_mode.assert_called_once()
        self.assertIn('data_name', result)


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
    @patch('msprobe.core.common.file_check.path_len_exceeds_limit', return_value=False)
    @patch('msprobe.core.common.file_check.change_mode')
    def test_maybe_save_overflow_data_and_check_overflow_times(self, mock_change_mode, mock_path_len_exceeds_limit, mock_save):
        self.processor.has_overflow = True
        self.processor.cached_tensors_and_file_paths = {'dummy_path': torch.tensor([1.0, 2.0, 3.0])}

        # First call should save the tensor and not raise an exception
        self.processor.maybe_save_overflow_data_and_check_overflow_times()
        mock_save.assert_called_once()
        mock_path_len_exceeds_limit.assert_called_once()
        mock_change_mode.assert_called_once()

        # Second call should raise an exception due to overflow nums limit
        with self.assertRaises(MsprobeException) as context:
            self.processor.maybe_save_overflow_data_and_check_overflow_times()

        self.assertEqual(str(context.exception), MsprobeException.OVERFLOW_NUMS_ERROR)

    def test_inc_and_check_overflow_times(self):
        self.processor.real_overflow_dump_times = 0
        self.processor.overflow_nums = 1
        self.processor.has_overflow = True

        # First increment should not raise an exception
        self.processor.inc_and_check_overflow_times()
        self.assertEqual(self.processor.real_overflow_dump_times, 1)

        # Second increment should raise an exception
        with self.assertRaises(MsprobeException) as context:
            self.processor.inc_and_check_overflow_times()

        self.assertEqual(str(context.exception), MsprobeException.OVERFLOW_NUMS_ERROR)

    @patch('os.getenv', return_value=Const.ENV_ENABLE)
    def test_overflow_debug_mode_enable(self, mock_getenv):
        result = self.processor.overflow_debug_mode_enable()
        self.assertTrue(result)
        mock_getenv.assert_called_once_with(OverflowConst.OVERFLOW_DEBUG_MODE_ENABLE, Const.ENV_DISABLE)

    @patch('numpy.isinf', return_value=True)
    @patch('numpy.isnan', return_value=False)
    def test_analyze_maybe_overflow_tensor(self, mock_isnan, mock_isinf):
        tensor_json = {'Max': float('inf'), 'Min': 1.0}
        self.processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertTrue(self.processor.has_overflow)

    @patch('msprobe.core.common.file_check.path_len_exceeds_limit', return_value=False)
    @patch.object(BaseDataProcessor, 'get_save_file_path', return_value=['test_api_name', 'test_api_name.0.forward.input.pt'])
    def test_analyze_tensor(self, mock_path_len_exceeds_limit, mock_get_save_file_path):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        suffix = 'suffix'
        with patch.object(PytorchDataProcessor, '_analyze_tensor', return_value={'Max': 3.0, 'Min': 1.0}) as mock_super_analyze_tensor:
            result = self.processor._analyze_tensor(tensor, suffix)
            mock_super_analyze_tensor.assert_called_once_with(tensor, suffix)
            self.assertIn('data_name', result)
            self.assertTrue(self.processor.has_overflow)

    @patch.object(PytorchDataProcessor, 'analyze_element', return_value=['mocked_result'])
    def test_analyze_backward(self, mock_analyze_element):
        module_io = ModuleBackwardInputsOutputs(grad_output=(1, 2), grad_input=(3, 4))
        self.config.data_mode = ["all"]
        result = self.processor.analyze_backward("test_backward", None, module_io)
        expected = {
            "test_backward": {
                "grad_input": ['mocked_result'],
                "grad_output": ['mocked_result']
            }
        }
        self.assertEqual(result, expected)

    @patch.object(PytorchDataProcessor, 'analyze_element', return_value=['mocked_result'])
    def test_analyze_forward(self, mock_analyze_element):
        module_io = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3}, output=(4, 5))
        self.config.data_mode = ["all"]
        result = self.processor.analyze_forward("test_forward", None, module_io)
        expected = {
            "test_forward": {
                "input_args": ['mocked_result'],
                "input_kwargs": ['mocked_result'],
                "output": ['mocked_result']
            }
        }
        self.assertEqual(result, expected)

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


# class TestKernelDumpDataProcessor(unittest.TestCase):

#     def setUp(self):
#         self.config = MagicMock()
#         self.config.is_forward_acl_dump = True
#         self.config.acl_config = "dummy_acl_config"
#         self.config.backward_input = {'test_module': 'dummy_path'}
#         self.data_writer = MagicMock()
#         self.processor = KernelDumpDataProcessor(self.config, self.data_writer)

#     @patch('torch_npu.npu.synchronize')
#     @patch('torch_npu.npu.init_dump')
#     @patch('torch_npu.npu.set_dump')
#     @patch('torch_npu.npu.finalize_dump')
#     def test_forward_acl_dump(self, mock_finalize_dump, mock_set_dump, mock_init_dump, mock_synchronize):
#         module = MagicMock()
#         module.forward = MagicMock(return_value=torch.tensor([1.0, 2.0, 3.0]))
#         module_io = MagicMock()
#         module_io.args = (1, 2)
#         module_io.kwargs = {'a': 3}
        
#         KernelDumpDataProcessor.forward_init_status = False
        
#         self.processor.forward_acl_dump('test_module', module, module_io)

#         mock_synchronize.assert_called()
#         mock_init_dump.assert_called_once_with()
#         mock_set_dump.assert_called_once_with("dummy_acl_config")
#         mock_finalize_dump.assert_called_once_with()
#         module.forward.assert_called_with(1, 2, a=3)

#     @patch('torch_npu.npu.synchronize')
#     @patch('torch_npu.npu.init_dump')
#     @patch('torch_npu.npu.set_dump')
#     @patch('torch_npu.npu.finalize_dump')
#     @patch('torch.load', return_value=torch.tensor([1.0, 2.0, 3.0]))
#     def test_dump_mode_backward_acl_dump(self, mock_load, mock_finalize_dump, mock_set_dump, mock_init_dump, mock_synchronize):
#         module = MagicMock()
#         module.forward = MagicMock(return_value=torch.tensor([1.0, 2.0, 3.0]))
#         module_io = MagicMock()
#         module_io.args = (1, 2)
#         module_io.kwargs = {'a': 3}
        
#         KernelDumpDataProcessor.forward_init_status = False
        
#         self.processor.dump_mode_backward_acl_dump('test_module', module, module_io)

#         mock_synchronize.assert_called()
#         mock_init_dump.assert_called_once_with()
#         mock_set_dump.assert_called_once_with("dummy_acl_config")
#         mock_finalize_dump.assert_called_once_with()
#         mock_load.assert_called_once_with('dummy_path')
#         module.forward.assert_called_with(1, 2, a=3)

#     def test_op_need_trigger(self):
#         self.assertTrue(self.processor.op_need_trigger('Tensor.__getitem__.'))
#         self.assertFalse(self.processor.op_need_trigger('SomeOtherOp'))

#     @patch.object(KernelDumpDataProcessor, 'forward_acl_dump')
#     @patch.object(KernelDumpDataProcessor, 'dump_mode_backward_acl_dump')
#     def test_analyze_forward(self, mock_dump_mode_backward_acl_dump, mock_forward_acl_dump):
#         self.processor.analyze_forward('test_module', MagicMock(), MagicMock())
#         mock_forward_acl_dump.assert_called_once()
#         mock_dump_mode_backward_acl_dump.assert_not_called()

#         self.config.is_forward_acl_dump = False
#         self.processor.analyze_forward('test_module', MagicMock(), MagicMock())
#         mock_dump_mode_backward_acl_dump.assert_called_once()
#         mock_forward_acl_dump.assert_called_once()  # 因为已经被调用过一次

#     @patch('torch.Tensor.backward')
#     def test_acl_backward_dump_status(self, mock_backward):
#         output = torch.tensor([1.0, 2.0, 3.0])
#         grad = torch.tensor([0.1, 0.1, 0.1])
#         self.assertTrue(self.processor.acl_backward_dump_status(output, grad, 'test_module'))
#         mock_backward.assert_called_once_with(grad, retain_graph=True)

#         output = [torch.tensor([1.0, 2.0, 3.0])]
#         self.assertTrue(self.processor.acl_backward_dump_status(output, grad, 'test_module'))
#         mock_backward.assert_called_with(grad, retain_graph=True)

#         output = [torch.tensor([1.0, 2.0, 3.0])]
#         self.assertFalse(self.processor.acl_backward_dump_status(output, grad, 'unknown_module'))

#     def tearDown(self):
#         KernelDumpDataProcessor.forward_init_status = False

# patcher.stop()
# class TestKernelDumpDataProcessor(unittest.TestCase):

#     def setUp(self):
#         self.config = MagicMock()
#         self.data_writer = MagicMock()
#         self.processor = KernelDumpDataProcessor(self.config, self.data_writer)

#     @patch('torch_npu.npu.synchronize')
#     @patch('torch_npu.npu.init_dump')
#     @patch('torch_npu.npu.set_dump')
#     @patch('torch_npu.npu.finalize_dump')
#     def test_forward_acl_dump(self, mock_finalize_dump, mock_set_dump, mock_init_dump, mock_synchronize):
#         module = MagicMock()
#         module.forward = MagicMock(return_value=torch.tensor([1.0, 2.0, 3.0]))
#         module_io = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3}, output=None)
#         self.processor.forward_acl_dump('test_module', module, module_io)
#         mock_synchronize.assert_called()
#         mock_init_dump.assert_called_once()
#         mock_set_dump.assert_called_once()
#         mock_finalize_dump.assert_called_once()

#     @patch('torch_npu.npu.synchronize')
#     @patch('torch_npu.npu.init_dump')
#     @patch('torch_npu.npu.set_dump')
#     @patch('torch_npu.npu.finalize_dump')
#     @patch('torch.load', return_value=torch.tensor([1.0, 2.0, 3.0]))
#     def test_dump_mode_backward_acl_dump(self, mock_load, mock_finalize_dump, mock_set_dump, mock_init_dump, mock_synchronize):
#         module = MagicMock()
#         module.forward = MagicMock(return_value=torch.tensor([1.0, 2.0, 3.0]))
#         module_io = ModuleForwardInputsOutputs(args=(1, 2), kwargs={'a': 3}, output=None)
#         self.config.backward_input = {'test_module': 'dummy_path'}
#         self.processor.dump_mode_backward_acl_dump('test_module', module, module_io)
#         mock_synchronize.assert_called()
#         mock_init_dump.assert_called_once()
#         mock_set_dump.assert_called_once()
#         mock_finalize_dump.assert_called_once()
#         mock_load.assert_called_once()

#     def test_op_need_trigger(self):
#         self.assertTrue(self.processor.op_need_trigger('Tensor.__getitem__.'))
#         self.assertFalse(self.processor.op_need_trigger('SomeOtherOp'))

if __name__ == '__main__':
    unittest.main()
