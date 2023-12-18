# coding=utf-8
import unittest
import numpy as np
import os
import copy
from api_accuracy_checker.run_ut.data_generate import *
from api_accuracy_checker.common.utils import get_json_contents

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
forward_file = os.path.join(base_dir, "../resources/forward.json")
forward_content = get_json_contents(forward_file)
for api_full_name, api_info_dict in forward_content.items():
    api_full_name = api_full_name
    api_info_dict = api_info_dict

max_value = 5.7421875
min_value = -5.125

class TestDataGenerateMethods(unittest.TestCase):
    def test_gen_api_params(self):
        api_info = copy.deepcopy(api_info_dict)
        args_params, kwargs_params = gen_api_params(api_info, True, None)
        max_diff = abs(args_params[0].max() - max_value)
        min_diff = abs(args_params[0].min() - min_value)
        self.assertEqual(len(args_params), 1)
        self.assertEqual(args_params[0].dtype, torch.float32)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(args_params[0].shape, torch.Size([2, 2560, 24, 24]))
        self.assertEqual(kwargs_params, {'inplace': False})

    def test_gen_args(self):
        args_result = gen_args(api_info_dict.get('args'))
        max_diff = abs(args_result[0].max() - max_value)
        min_diff = abs(args_result[0].min() - min_value)
        self.assertEqual(len(args_result), 1)
        self.assertEqual(args_result[0].dtype, torch.float32)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(args_result[0].shape, torch.Size([2, 2560, 24, 24]))

    def test_gen_data(self):
        data = gen_data(api_info_dict.get('args')[0], True, None)
        max_diff = abs(data.max() - max_value)
        min_diff = abs(data.min() - min_value)
        self.assertEqual(data.dtype, torch.float32)
        self.assertEqual(data.requires_grad, True)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(data.shape, torch.Size([2, 2560, 24, 24]))

    def test_gen_kwargs(self):
        api_info = copy.deepcopy(api_info_dict)
        kwargs_params = gen_kwargs(api_info, None)
        self.assertEqual(kwargs_params, {'inplace': False})
        
    def test_gen_kwargs_device(self):
        k_dict = {"kwargs": {"device": {"type": "torch.device", "value": "cpu"}}}
        kwargs_params = gen_kwargs(k_dict, None)
        self.assertEqual(str(kwargs_params), "{'device': device(type='cpu')}")
    
    def test_gen_kwargs_1(self):
        k_dict = {"device": {"type": "torch.device", "value": "cpu"}}
        for key, value in k_dict.items():
            gen_torch_kwargs(k_dict, key, value)
        self.assertEqual(str(k_dict), "{'device': device(type='cpu')}")
        
    def test_gen_kwargs_2(self):
        k_dict = {"inplace": {"type": "bool", "value": "False"}}
        for key, value in k_dict.items():
            gen_torch_kwargs(k_dict, key, value)
        self.assertEqual(k_dict, {'inplace': False})
    
    def test_gen_random_tensor(self):
        data = gen_random_tensor(api_info_dict.get('args')[0], None)
        max_diff = abs(data.max() - max_value)
        min_diff = abs(data.min() - min_value)
        self.assertEqual(data.dtype, torch.float32)
        self.assertEqual(data.requires_grad, False)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(data.shape, torch.Size([2, 2560, 24, 24]))
        
    def test_gen_common_tensor(self):
        info = api_info_dict.get('args')[0]
        low, high = info.get('Min'), info.get('Max')
        data_dtype = info.get('dtype')
        shape = tuple(info.get('shape'))
        data = gen_common_tensor(low, high, shape, data_dtype, None)
        max_diff = abs(data.max() - max_value)
        min_diff = abs(data.min() - min_value)
        self.assertEqual(data.dtype, torch.float32)
        self.assertEqual(data.requires_grad, False)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(data.shape, torch.Size([2, 2560, 24, 24]))
        
    def test_gen_bool_tensor(self):
        info = {"type": "torch.Tensor", "dtype": "torch.bool", "shape": [1, 1, 160, 256], \
            "Max": 1, "Min": 0, "requires_grad": False}
        low, high = info.get("Min"), info.get("Max")
        shape = tuple(info.get("shape"))
        data = gen_bool_tensor(low, high, shape)
        self.assertEqual(data.shape, torch.Size([1, 1, 160, 256]))
        self.assertEqual(data.dtype, torch.bool)
