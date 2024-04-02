import os
import shutil
import unittest
import torch
import numpy as np
from api_accuracy_checker.dump.api_info import APIInfo, ForwardAPIInfo, BackwardAPIInfo, transfer_types, \
    get_tensor_extremum, get_type_name, is_builtin_class, analyze_device_in_kwargs, analyze_dtype_in_kwargs
from api_accuracy_checker.common.config import msCheckerConfig


class TestAPIInfo(unittest.TestCase):
    def setUp(self):
        if os.path.exists('./step-1'):
            shutil.rmtree('./step-1')
        self.api = APIInfo("test_api", APIInfo.get_full_save_path("./", "forward_real_data", True), True)

    def test_analyze_element(self):
        element = [1, 2, 3]
        result = self.api.analyze_element(element)
        self.assertEqual(result,
                         [{'type': 'int', 'value': 1}, {'type': 'int', 'value': 2}, {'type': 'int', 'value': 3}])

    def test_analyze_tensor(self):
        tensor = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
        result = self.api._analyze_tensor(tensor)
        self.assertEqual(result.get('type'), 'torch.Tensor')
        self.assertTrue(result.get('requires_grad'))
        datapath = result.get('datapath')
        self.assertTrue(datapath.startswith('forward_real_data') or datapath.startswith('backward_real_data'))

    def test_analyze_builtin(self):
        arg = slice(1, 10, 2)
        result = self.api._analyze_builtin(arg)
        self.assertEqual(result, {'type': 'slice', 'value': [1, 10, 2]})

    def test_transfer_types(self):
        data = 10
        dtype = 'int'
        result = transfer_types(data, dtype)
        self.assertEqual(result, 10)

    def test_is_builtin_class(self):
        element = 10
        result = is_builtin_class(element)
        self.assertTrue(result)

    def test_analyze_device_in_kwargs(self):
        element = torch.device('cuda:0')
        result = analyze_device_in_kwargs(element)
        self.assertEqual(result, {'type': 'torch.device', 'value': 'cuda:0'})

    def test_analyze_dtype_in_kwargs(self):
        element = torch.float32
        result = analyze_dtype_in_kwargs(element)
        self.assertEqual(result, {'type': 'torch.dtype', 'value': 'torch.float32'})

    def test_get_tensor_extremum(self):
        data = torch.tensor([1, 2, 3])
        result_max, result_max_origin = get_tensor_extremum(data, 'max')
        result_min, result_min_origin = get_tensor_extremum(data, 'min')
        self.assertEqual(result_max, 3)
        self.assertEqual(result_min, 1)
        self.assertEqual(result_max_origin, 3)
        self.assertEqual(result_min_origin, 1)
        
        data = torch.tensor([1, float("inf"), 2, 3])
        result_max, result_max_origin = get_tensor_extremum(data, 'max')
        result_min, result_min_origin = get_tensor_extremum(data, 'min')
        self.assertEqual(result_max, 3)
        self.assertEqual(result_min, 1)
        self.assertEqual(result_max_origin, float("inf"))
        self.assertEqual(result_min_origin, 1)

        data = torch.tensor([1, float("-inf"), 2, 3])
        result_max, result_max_origin = get_tensor_extremum(data, 'max')
        result_min, result_min_origin = get_tensor_extremum(data, 'min')
        self.assertEqual(result_max, 3)
        self.assertEqual(result_min, 1)
        self.assertEqual(result_max_origin, 3)
        self.assertEqual(result_min_origin, float("-inf"))

        data = torch.tensor([1, float("inf"), float("nan"), 3])
        result_max, result_max_origin = get_tensor_extremum(data, 'max')
        result_min, result_min_origin = get_tensor_extremum(data, 'min')
        self.assertEqual(result_max, 3)
        self.assertEqual(result_min, 1)
        self.assertEqual(bool(np.isnan(result_max_origin)), True)
        self.assertEqual(bool(np.isnan(result_min_origin)), True)

        data = torch.tensor([float("inf"), float("nan")])
        result_max, result_max_origin = get_tensor_extremum(data, 'max')
        result_min, result_min_origin = get_tensor_extremum(data, 'min')
        self.assertEqual(result_max, float("inf"))
        self.assertEqual(result_min, float("inf"))
        self.assertEqual(bool(np.isnan(result_max_origin)), True)
        self.assertEqual(bool(np.isnan(result_min_origin)), True)

        data = torch.tensor([float("nan"), float("nan")])
        result_max, result_max_origin = get_tensor_extremum(data, 'max')
        result_min, result_min_origin = get_tensor_extremum(data, 'min')
        self.assertEqual(bool(np.isnan(result_max)), True)
        self.assertEqual(bool(np.isnan(result_min)), True)
        self.assertEqual(bool(np.isnan(result_max_origin)), True)
        self.assertEqual(bool(np.isnan(result_min_origin)), True)

    def test_get_type_name(self):
        name = "<class 'int'>"
        result = get_type_name(name)
        self.assertEqual(result, 'int')

    def test_ForwardAPIInfo(self):
        forward_api_info = ForwardAPIInfo("test_forward_api", [1, 2, 3], {"a": 1, "b": 2})
        self.assertEqual(forward_api_info.api_name, "test_forward_api")
        self.assertEqual(forward_api_info.save_path,
                         APIInfo.get_full_save_path(msCheckerConfig.dump_path, 'forward_real_data', True))
        self.assertEqual(forward_api_info.api_info_struct, {"test_forward_api": {
            "args": [{'type': 'int', 'value': 1}, {'type': 'int', 'value': 2}, {'type': 'int', 'value': 3}, ],
            "kwargs": {'a': {'type': 'int', 'value': 1}, 'b': {'type': 'int', 'value': 2}}}})

    def test_BackwardAPIInfo(self):
        backward_api_info = BackwardAPIInfo("test_backward_api", [1, 2, 3])
        self.assertEqual(backward_api_info.api_name, "test_backward_api")
        self.assertEqual(backward_api_info.save_path,
                         APIInfo.get_full_save_path(msCheckerConfig.dump_path, 'backward_real_data', True))
        self.assertEqual(backward_api_info.grad_info_struct, {
            "test_backward_api": [{'type': 'int', 'value': 1}, {'type': 'int', 'value': 2},
                                  {'type': 'int', 'value': 3}]})

    def tearDown(self):
        if os.path.exists('./step-1'):
            shutil.rmtree('./step-1')
