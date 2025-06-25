# coding=utf-8
import os
import unittest
from unittest.mock import patch
import copy
import math
import numpy as np
import torch

from msprobe.pytorch.api_accuracy_checker.run_ut.data_generate import *
from msprobe.core.common.file_utils import get_json_contents, create_directory
from msprobe.pytorch.common.utils import save_pt

base_dir = os.path.dirname(os.path.realpath(__file__))
forward_file = os.path.join(base_dir, "forward.json")
forward_content = get_json_contents(forward_file)
for key, value in forward_content.items():
    api_full_name = key
    api_info_dict = value

max_value = 1.3945078125
min_value = -1.444359375


class TestDataGenerateMethods(unittest.TestCase):
    
    def setUp(self):
        tensor = torch.tensor(1.0, dtype=torch.int32)
        save_path = "temp_save_path"
        create_directory(save_path)
        self.save_path = os.path.realpath(save_path)
        self.tensor_path = os.path.join(self.save_path, "tensor.pt")
        save_pt(tensor, self.tensor_path)
        self.npy_path = os.path.join(self.save_path, "npy.npy")
        np.save(self.npy_path, np.array(1.0))
        self.txt_path = os.path.join(self.save_path, "txt.txt")
        with open(self.txt_path, 'w') as f:
            f.write("1.0")

    def tearDown(self):
        for file in os.listdir(self.save_path):
            os.remove(os.path.join(self.save_path, file))
        os.rmdir(self.save_path)
    
    def test_gen_api_params(self):
        api_info = copy.deepcopy(api_info_dict)
        args_params, kwargs_params, output_dtype = gen_api_params(api_info, "conv2d", True, None, None)
        max_diff = abs(args_params[0].max() - max_value)
        min_diff = abs(args_params[0].min() - min_value)
        self.assertEqual(len(args_params), 2)
        self.assertEqual(args_params[0].dtype, torch.float16)
        self.assertEqual(args_params[1], 2)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(args_params[0].shape, torch.Size([2048, 2, 1, 256]))
        self.assertEqual(kwargs_params, {'dim': -1})
        self.assertEqual(output_dtype, torch.float16)

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
        info = {'type': 'torch.Tensor', 'datapath': "tensor.pt"}
        api_name = "test_api"
        data = gen_data(info, api_name, need_grad=True, convert_type=None, real_data_path=self.save_path)
        self.assertIsInstance(data, torch.Tensor)
    
    def test_gen_data_fp32_to_hf32_to_fp32(self):
        info = {'type': "torch.Tensor", 'Min': 0, 'Max': 1, 'dtype': "torch.float32", 'shape': (1, 2)}
        api_name = "conv2d"
        input_tensor = gen_random_tensor(info, convert_type=None)
        origin_result = gen_data(info, api_name, False, None)
        expect_result = fp32_to_hf32_to_fp32(input_tensor)
        self.assertTrue(torch.allclose(origin_result, expect_result, atol=1e-4))

    def test_gen_data_numpy_data_type(self):
        info = {'type': 'numpy.float64', 'value': 3.14}
        api_name = "test_api"
        data = gen_data(info, api_name, need_grad=False, convert_type=None)
        self.assertEqual(data, 3.14)
        
        info = {'Min': 0, 'Max': 1, 'dtype': 'numpy.unknown', 'shape': (2, 3)}
        with self.assertRaises(Exception) as context:
            data = gen_data(info, api_name, need_grad=False, convert_type=None)

    def test_gen_data_special_data_types(self):
        info = {'type': 'torch.Size', 'value': (2, 3)}
        api_name = "test_api"
        data = gen_data(info, api_name, need_grad=False, convert_type=None)
        self.assertEqual(data, torch.Size([2, 3]))
        
        info = {'type': 'slice', 'value': [1, 10, 2]}
        api_name = "test_api"
        data = gen_data(info, api_name, need_grad=False, convert_type=None)
        self.assertEqual(data, slice(1, 10, 2))
        
        info = {'type': 'ellipsis', 'value': '...'}
        api_name = "test_api"
        data = gen_data(info, api_name, need_grad=False, convert_type=None)
        self.assertEqual(data, ...)
        
    def test_gen_real_tensor_load_pt(self):
        data_path = self.tensor_path 
        data = gen_real_tensor(data_path, convert_type=None)
        self.assertIsInstance(data, torch.Tensor)

    def test_load_npy_file(self):
        data_path = self.npy_path
        data = gen_real_tensor(data_path, convert_type=None)
        self.assertIsInstance(data, torch.Tensor)

    def test_gen_real_tensor_unsupported_file_format(self):
        data_path = self.txt_path
        with self.assertRaises(CompareException) as context:
            gen_real_tensor(data_path, convert_type=None)
        self.assertEqual(context.exception.code, CompareException.INVALID_FILE_ERROR)

    def test_gen_real_tensor_data_type_conversion(self):
        data_path = self.tensor_path 
        data = gen_real_tensor(data_path, convert_type='int32_to_int64')
        self.assertEqual(data.dtype, torch.int64)

    def test_gen_random_tensor_gen_bool_tensor(self):
        info = {'Min': 0, 'Max': 1, 'dtype': "torch.bool", 'shape': (1, 2)}
        data = gen_random_tensor(info, convert_type=None)
        self.assertEqual(data.dtype, torch.bool)
    
    def test_gen_random_tensor_gen_cat(self):
        info = {'Min': None, 'Max': None, 'dtype': "torch.float32", 'shape': (1, 0, 256)}
        data = gen_random_tensor(info, None)
        self.assertEqual(data.dtype, torch.float32)
        self.assertEqual(data.shape, torch.Size([1, 0, 256]))

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
        self.assertTrue(torch.allclose(fp32_to_hf32_to_fp32(input_tensor), input_hf32, atol=1e-4))

    def test_gen_kwargs(self):
        api_info = copy.deepcopy(api_info_dict)
        kwargs_params = gen_kwargs(api_info, None)
        self.assertEqual(kwargs_params, {'dim': -1})
    
    def test_gen_kwargs_fa_special_sparse_mode(self):
        api_info = {"input_kwargs": {"atten_mask": {"type": "torch.Tensor", "shape": [2048, 2048]}, 
                                     "sparse_mode": {"type": "int", "value": 3}}}
        api_name = "npu_fusion_attention"
        kwargs_params = gen_kwargs(api_info, api_name, None, None)
        
        # 分别验证 kwargs_params 的每个键值
        self.assertIn('atten_mask', kwargs_params)
        self.assertIn('sparse_mode', kwargs_params)
        
        # 验证 atten_mask 的属性
        expected_mask = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(torch.bool)
        self.assertTrue(torch.equal(kwargs_params['atten_mask'], expected_mask))
        
        # 验证 sparse_mode 的值
        self.assertEqual(kwargs_params['sparse_mode'], 3)

    def test_gen_kwargs_2(self):
        k_dict = {"dtype": {"type": "torch.dtype", "value": "torch.float16"}}
        for key, value in k_dict.items():
            gen_torch_kwargs(k_dict, key, value)
        self.assertEqual(k_dict, {'dtype': torch.float16})
        
    def test_gen_kwargs_gen_list_kwargs(self):
        api_info = {"input_kwargs": {"key": [{'type': 'int', 'value': 1.0}]}}
        api_name = "test_api"
        convert_type = None
        real_data_path = None
        kwargs = gen_kwargs(api_info, api_name, convert_type, real_data_path)
        self.assertEqual(kwargs["key"], [1.0])
        
    def test_gen_kwargs_none_kwargs(self):
        api_info = {"input_kwargs": {"key": None}}
        api_name = "test_api"
        convert_type = None
        real_data_path = None
        kwargs = gen_kwargs(api_info, api_name, convert_type, real_data_path)
        self.assertIsNone(kwargs["key"])

    def test_gen_kwargs_tensor_kwargs(self):
        api_info = {"input_kwargs": {"key": {"type": "torch.Tensor", 'Min': 0, 'Max': 1, 
                                             'dtype': "torch.float16", 'shape': (1, 2)}}}
        api_name = "test_api"
        convert_type = None
        real_data_path = None
        kwargs = gen_kwargs(api_info, api_name, convert_type, real_data_path)
        print(kwargs['key'])
        self.assertIsInstance(kwargs["key"], torch.Tensor)
        
    def test_torch_kwargs(self):
        api_info = {"input_kwargs": {"key": {"type": "torch.Size", 'value': (2, 3)}}}
        api_name = "test_api"
        convert_type = None
        real_data_path = None
        kwargs = gen_kwargs(api_info, api_name, convert_type, real_data_path)
        print(kwargs['key'])
        self.assertEqual(kwargs["key"], (2, 3))

    def test_gen_list_kwargs(self):
        kwargs_item_value = [{'type': 'torch.float32', 'value': 1.0}, {'type': 'torch.int32', 'value': 2}]
        api_name = "test_api"
        convert_type = None
        real_data_path = None
        expect_return_value = [torch.tensor(1.0), torch.tensor(2)]
        result = gen_list_kwargs(kwargs_item_value, api_name, convert_type, real_data_path)
        self.assertEqual(result, expect_return_value)
        
        kwargs_item_value = [{'type': 'torch.Size', 'value': (2, 3)}]
        result = gen_list_kwargs(kwargs_item_value, api_name, convert_type, real_data_path)
        self.assertEqual(result[0], torch.Size([2, 3]))
        
        kwargs_item_value = [{'type': 'torch.Tensor', 'Min': 0, 'Max': 1, 'dtype': "torch.float16", 'shape': (1, 2)}]
        result = gen_list_kwargs(kwargs_item_value, api_name, convert_type, real_data_path)
        self.assertIsInstance(result[0], torch.Tensor)
        
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
        data_dtype = 'torch.int32'
        convert_type = 'int32_to_int64'
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, convert_type)
        self.assertEqual(tensor.dtype, torch.int64)

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
        self.assertTrue(tensor.max() == float('inf'))
        self.assertTrue(tensor.min() == float('-inf'))

        low_info = [float('inf'), float('inf')]
        high_info = [float('inf'), float('inf')]
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        self.assertTrue(tensor.max() == float('inf'))
        self.assertTrue(tensor.min() == float('inf'))

        low_info = [1, float('inf')]
        high_info = [2, float('inf')]
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        self.assertTrue(tensor.max() == float('inf'))

        low_info = [1, float('-inf')]
        high_info = [2, float('-inf')]
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        self.assertTrue(tensor.min() == float('-inf'))

        low_info = [1, float('nan')]
        high_info = [2, float('nan')]
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        self.assertTrue(math.isnan(tensor.max()) and math.isnan(tensor.min()))

        low_info = [float('nan'), float('nan')]
        high_info = [float('nan'), float('nan')]
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        self.assertTrue(math.isnan(tensor.max()) and math.isnan(tensor.min()))

        shape = (0, 0)
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        self.assertEqual(tensor.numel(), 0)

        shape = (1, 2)
        low_info = [2, float('nan')]
        high_info = [2, float('nan')]
        tensor = gen_common_tensor(low_info, high_info, shape, data_dtype, None)
        self.assertTrue(math.isnan(tensor.max()) and math.isnan(tensor.min()))

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
        self.assertEqual(data.dtype, torch.bool)

    def test_gen_api_params(self):
        api_info = {"input_args": [], "input_kwargs": {}}
        api_name = "test_api"
        need_grad = True
        convert_type = "unsupported_type"
        real_data_path = None
        with self.assertRaises(CompareException) as context:
            gen_api_params(api_info, api_name, need_grad, convert_type, real_data_path)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        
        convert_type = None
        api_info = {"input_args": None, "input_kwargs": {}}
        with patch('msprobe.pytorch.common.log.logger.warning') as mock_logger:
            result_args, result_kwargs, _ = gen_api_params(api_info, api_name, need_grad, convert_type, real_data_path)
            self.assertEqual(result_args, [])
            mock_logger.assert_called_once_with(f'Warning: No args in {api_info} ')
