# coding=utf-8
import os
import unittest
from unittest.mock import patch
import copy
import math
import numpy as np

from msprobe.pytorch.api_accuracy_checker.run_ut.data_generate import *
from msprobe.core.common.file_utils import get_json_contents

base_dir = os.path.dirname(os.path.realpath(__file__))
forward_file = os.path.join(base_dir, "forward.json")
forward_content = get_json_contents(forward_file)
for key, value in forward_content.items():
    api_full_name = key
    api_info_dict = value

max_value = 1.3945078125
min_value = -1.444359375


class TestDataGenerateMethods(unittest.TestCase):
    def test_gen_api_params(self):
        api_info = copy.deepcopy(api_info_dict)
        args_params, kwargs_params = gen_api_params(api_info, "conv2d", True, None, None)
        max_diff = abs(args_params[0].max() - max_value)
        min_diff = abs(args_params[0].min() - min_value)
        self.assertEqual(len(args_params), 2)
        self.assertEqual(args_params[0].dtype, torch.float16)
        self.assertEqual(args_params[1], 2)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(args_params[0].shape, torch.Size([2048, 2, 1, 256]))
        self.assertEqual(kwargs_params, {'dim': -1})

    def test_gen_args(self):
        func_options = {}
        api_name = "conv2d"
        args_result = gen_args(api_info_dict.get('input_args'), "conv2d", func_options)
        max_diff = abs(args_result[0].max() - max_value)
        min_diff = abs(args_result[0].min() - min_value)
        self.assertEqual(len(args_result), 2)
        self.assertEqual(args_result[0].dtype, torch.float16)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(args_result[0].shape, torch.Size([2048, 2, 1, 256]))

        args_info = [object()]
        with self.assertRaises(NotImplementedError):
            gen_args(args_info, api_name, func_options)

        args_info = [None]
        result = gen_args(args_info, api_name, func_options)
        self.assertIsNone(result[0])

    def test_gen_data(self):
        data = gen_data(api_info_dict.get('input_args')[0], "conv2d", True, None, None)
        max_diff = abs(data.max() - max_value)
        min_diff = abs(data.min() - min_value)
        self.assertEqual(data.dtype, torch.float16)
        self.assertEqual(data.requires_grad, True)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(data.shape, torch.Size([2048, 2, 1, 256]))
        
    def test_gen_data_gen_real_data(self):
        with patch('msprobe.pytorch.api_accuracy_checker.run_ut.data_generate.gen_real_tensor') as mock_gen_real:
            mock_gen_real.return_value = torch.tensor(1.0)
            info = {'type': 'torch.float32', 'datapath': 'path/to/data.pt'}
            api_name = "test_api"
            data = gen_data(info, api_name, need_grad=True, convert_type=None)
            self.assertTrue(mock_gen_real.called)

    def test_gen_data_numpy_data_type(self):
        info = {'type': 'numpy.float64', 'value': 3.14}
        api_name = "test_api"
        data = gen_data(info, api_name, need_grad=False, convert_type=None)
        self.assertIsInstance(data, np.ndarray)

    def test_gen_data_special_data_types(self):
        info = {'type': 'torch.Size', 'value': (2, 3)}
        api_name = "test_api"
        data = gen_data(info, api_name, need_grad=False, convert_type=None)
        self.assertEqual(data, torch.Size(2, 3))
        
        info = {'type': 'slice', 'value': [1, 10, 2]}
        api_name = "test_api"
        data = gen_data(info, api_name, need_grad=False, convert_type=None)
        self.assertEqual(data, slice(1, 10, 2))
        
        info = {'type': 'ellipsis', 'value': '...'}
        api_name = "test_api"
        data = gen_data(info, api_name, need_grad=False, convert_type=None)
        self.assertEqual(data, ...)
        
    def test_gen_real_tensor_load_pt(self):
        with patch('msprobe.pytorch.common.utils.load_pt') as mock_load_pt:
            mock_load_pt.return_value = torch.tensor(1.0)
            data_path = '/path/to/data.pt'
            data = gen_real_tensor(data_path, convert_type=None)
            self.assertTrue(mock_load_pt.called)
            self.assertIsInstance(data, torch.Tensor)

    def test_load_npy_file(self):
        with patch('msprobe.core.common.file_utils.load_npy') as mock_load_npy:
            mock_npy_data = np.array([1.0])
            mock_load_npy.return_value = mock_npy_data
            data_path = '/path/to/data.npy'
            data = gen_real_tensor(data_path, convert_type=None)
            self.assertTrue(mock_load_npy.called)
            self.assertIsInstance(data, torch.Tensor)

    def test_gen_real_tensor_unsupported_file_format(self):
        data_path = '/path/to/data.txt'
        with self.assertRaises(CompareException) as context:
            gen_real_tensor(data_path, convert_type=None)
        self.assertEqual(context.exception.code, CompareException.INVALID_FILE_ERROR)

    def test_gen_real_tensor_data_type_conversion(self):
        with patch('msprobe.pytorch.api_accuracy_checker.common.utils.get_module_and_atttribute_name') \
            as mock_get_module_and_attribute_name:
            with patch('msprobe.pytorch.api_accuracy_checker.common.utils.get_attribute') as mock_get_attribute:
                mock_get_module_and_attribute_name.return_value = ('torch', 'float32')
                mock_get_attribute.return_value = torch.float32
                data_path = '/path/to/data.pt'
                with patch('msprobe.pytorch.common.utils.load_pt') as mock_load_pt:
                    mock_tensor = torch.tensor(1.0, dtype=torch.float64)
                    mock_load_pt.return_value = mock_tensor
                    data = gen_real_tensor(data_path, convert_type='float32')
                    self.assertEqual(data.dtype, torch.float32)

    def test_gen_random_tensor_invalid_boundaries(self):
        info = {'Min': 'a', 'Max': 'b', 'dtype': 'torch.float32', 'shape': (2, 3)}
        with self.assertRaises(CompareException) as context:
            gen_random_tensor(info, convert_type=None)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)

    def test_gen_random_tensor_gen_bool_tensor(self):
        info = {'Min': 0, 'Max': 1, 'dtype': "torch.bool", 'shape': (2, 3)}
        with patch('your_module.gen_bool_tensor') as mock_gen_bool:
            mock_gen_bool.return_value = torch.tensor([True, False])
            data = gen_random_tensor(info, convert_type=None)
            self.assertTrue(mock_gen_bool.called)
            self.assertEqual(data.dtype, torch.bool)

    def test_gen_random_tensor(self):
        data = gen_random_tensor(api_info_dict.get('input_args')[0], None)
        max_diff = abs(data.max() - max_value)
        min_diff = abs(data.min() - min_value)
        self.assertEqual(data.dtype, torch.float16)
        self.assertEqual(data.requires_grad, False)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(data.shape, torch.Size([2048, 2, 1, 256]))
    
    def test_fp32_to_hf32_to_fp32(self):
        input_tensor = torch.rand(3, dtype=torch.float32)
        input_np = input_tensor.detach().numpy()
        input_int = input_np.view(numpy.int32)
        input_int = numpy.right_shift(numpy.right_shift(input_int, 11) + 1, 1)
        input_int = numpy.left_shift(input_int, 12)
        input_fp32 = input_int.view(numpy.float32)
        input_hf32 = torch.from_numpy(input_fp32)
        self.assertEqual(fp32_to_hf32_to_fp32(input_tensor), input_hf32)

    def test_gen_kwargs(self):
        api_info = copy.deepcopy(api_info_dict)
        kwargs_params = gen_kwargs(api_info, None)
        self.assertEqual(kwargs_params, {'dim': -1})

    def test_gen_kwargs_2(self):
        k_dict = {"dtype": {"type": "torch.dtype", "value": "torch.float16"}}
        for key, value in k_dict.items():
            gen_torch_kwargs(k_dict, key, value)
        self.assertEqual(k_dict, {'dtype': torch.float16})

    def test_gen_list_kwargs(self):
        kwargs_item_value = [{'type': 'torch.float32', 'value': 1.0}, {'type': 'torch.int32', 'value': 2}]
        api_name = "test_api"
        convert_type = None
        real_data_path = None
        with patch('msprobe.pytorch.api_accuracy_checker.run_ut.data_generate.gen_data') as mock_gen_data:
            mock_gen_data.side_effect = [torch.tensor(1.0), torch.tensor(2)]
            result = gen_list_kwargs(kwargs_item_value, api_name, convert_type, real_data_path)
            self.assertIsInstance(result[0], torch.Tensor)
            self.assertIsInstance(result[1], torch.Tensor)
        
        kwargs_item_value = [{'type': 'torch.Size', 'value': (2, 3)}]
        result = gen_list_kwargs(kwargs_item_value, api_name, convert_type, real_data_path)
        self.assertEqual(result[0], torch.Size(2, 3))
        
        kwargs_item_value = [{'type': 'str', 'value': 'hello'}, {'type': 'int', 'value': 42}]
        result = gen_list_kwargs(kwargs_item_value, api_name, convert_type, real_data_path)
        self.assertEqual(result[0], 'hello')
        self.assertEqual(result[1], 42)

    def test_gen_common_tensor(self):
        info = api_info_dict.get('input_args')[0]
        low, high = info.get('Min'), info.get('Max')
        low_origin, high_origin = info.get('Min_origin'), info.get('Max_origin')
        low_info = [low, low_origin]
        high_info = [high, high_origin]
        data_dtype = info.get('dtype')
        shape = tuple(info.get('shape'))
        data = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        max_diff = abs(data.max() - max_value)
        min_diff = abs(data.min() - min_value)
        self.assertEqual(data.dtype, torch.float16)
        self.assertEqual(data.requires_grad, False)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(data.shape, torch.Size([2048, 2, 1, 256]))
        
    def test_gen_common_tensor_data_type_conversion(self):
        low_info = [0.0, 0.0]
        high_info = [1.0, 1.0]
        shape = (3, 3)
        data_dtype = 'torch.float32'
        convert_type = 'torch.float64'
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, convert_type)
        self.assertEqual(tensor.dtype, torch.float64)

    def test_gen_common_tensor_unsupported_dtype(self):
        low_info = [0.0, 0.0]
        high_info = [1.0, 1.0]
        shape = (3, 3)
        data_dtype = 'unsupported_dtype'
        with self.assertRaises(NotImplementedError):
            gen_common_tensor(low_info, high_info, shape, data_dtype, None)
            
    def test_gen_common_tensor_integer_tensor_generation(self):
        low_info = [0, 0]
        high_info = [5, 5]
        shape = (3, 3)
        data_dtype = 'torch.int32'
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.int32)

    def test_gen_common_tensor_boundary_values(self):
        low_info = [1, float('-inf')]
        high_info = [2, float('inf')]
        shape = (3, 3)
        data_dtype = 'torch.float32'
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        self.assertTrue(tensor.max == float('inf') and tensor.min == float('-inf'))

        low_info = [float('inf'), float('inf')]
        high_info = [float('inf'), float('inf')]
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        self.assertTrue(math.isnan(tensor.max) and math.isnan(tensor.min))
        
        low_info = [float('-inf'), float('-inf')]
        high_info = [float('-inf'), float('-inf')]
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        self.assertTrue(math.isnan(tensor.max) and math.isnan(tensor.min))

        low_info = [float('nan'), float('nan')]
        high_info = [float('nan'), float('nan')]
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        self.assertTrue(math.isnan(tensor.max) and math.isnan(tensor.min))
        
        shape = 0
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        self.assertTrue(tensor == torch.tensor([]))

    def test_gen_bool_tensor(self):
        info = {"type": "torch.Tensor", "dtype": "torch.bool", "shape": [1, 1, 160, 256], "Max": 1, "Min": 0,
                "requires_grad": False}
        low, high = info.get("Min"), info.get("Max")
        shape = tuple(info.get("shape"))
        data = gen_bool_tensor(low, high, shape)
        self.assertEqual(data.shape, torch.Size([1, 1, 160, 256]))
        self.assertEqual(data.dtype, torch.bool)
        
        low, high = 1, 0
        shape = (1, 2)
        data = gen_bool_tensor(low, high, shape)
        self.assertEqual(data, torch.tensor([[False, True]]))

    def test_gen_api_params(self):
        api_info = {"input_args": [], "input_kwargs": {}}
        api_name = "test_api"
        need_grad = True
        convert_type = "unsupported_type"
        real_data_path = None
        with self.assertRaises(CompareException) as context:
            gen_api_params(api_info, api_name, need_grad, convert_type, real_data_path)
        self.assertEqual(context.exception.error_code, CompareException.INVALID_PARAM_ERROR)
        
        api_info = {"input_args": None, "input_kwargs": {}}
        with patch('msprobe.pytorch.common.log.logger.warning') as mock_logger:
            result_args, result_kwargs = gen_api_params(api_info, api_name, need_grad, convert_type, real_data_path)
            self.assertEqual(result_args, [])
            mock_logger.assert_called_once_with(f'Warning: No args in {api_info} ')
