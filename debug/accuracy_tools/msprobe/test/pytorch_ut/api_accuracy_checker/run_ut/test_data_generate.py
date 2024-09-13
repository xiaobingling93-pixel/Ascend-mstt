# coding=utf-8
import os
import unittest
import copy

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
        args_result = gen_args(api_info_dict.get('input_args'), "conv2d")
        max_diff = abs(args_result[0].max() - max_value)
        min_diff = abs(args_result[0].min() - min_value)
        self.assertEqual(len(args_result), 2)
        self.assertEqual(args_result[0].dtype, torch.float16)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(args_result[0].shape, torch.Size([2048, 2, 1, 256]))

    def test_gen_data(self):
        data = gen_data(api_info_dict.get('input_args')[0], "conv2d", True, None, None)
        max_diff = abs(data.max() - max_value)
        min_diff = abs(data.min() - min_value)
        self.assertEqual(data.dtype, torch.float16)
        self.assertEqual(data.requires_grad, True)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(data.shape, torch.Size([2048, 2, 1, 256]))

    def test_gen_kwargs(self):
        api_info = copy.deepcopy(api_info_dict)
        kwargs_params = gen_kwargs(api_info, None)
        self.assertEqual(kwargs_params, {'dim': -1})

    def test_gen_kwargs_2(self):
        k_dict = {"inplace": {"type": "bool", "value": "False"}}
        for key, value in k_dict.items():
            gen_torch_kwargs(k_dict, key, value)
        self.assertEqual(k_dict, {'inplace': False})

    def test_gen_random_tensor(self):
        data = gen_random_tensor(api_info_dict.get('input_args')[0], None)
        max_diff = abs(data.max() - max_value)
        min_diff = abs(data.min() - min_value)
        self.assertEqual(data.dtype, torch.float16)
        self.assertEqual(data.requires_grad, False)
        self.assertLessEqual(max_diff, 0.001)
        self.assertLessEqual(min_diff, 0.001)
        self.assertEqual(data.shape, torch.Size([2048, 2, 1, 256]))

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

    def test_gen_bool_tensor(self):
        info = {"type": "torch.Tensor", "dtype": "torch.bool", "shape": [1, 1, 160, 256], "Max": 1, "Min": 0,
                "requires_grad": False}
        low, high = info.get("Min"), info.get("Max")
        shape = tuple(info.get("shape"))
        data = gen_bool_tensor(low, high, shape)
        self.assertEqual(data.shape, torch.Size([1, 1, 160, 256]))
        self.assertEqual(data.dtype, torch.bool)
