import unittest
import torch
import os
import shutil
from api_accuracy_checker.common.base_api import BaseAPIInfo

class TestBaseAPI(unittest.TestCase):
    def setUp(self):
        if os.path.exists('./forward'):
            shutil.rmtree('./forward')
        os.makedirs('./forward', mode=0o755)
        self.api = BaseAPIInfo("test_api", True, True, "./", "forward", "backward")

    def test_analyze_element(self):
        element = [1, 2, 3]
        result = self.api.analyze_element(element)
        self.assertEqual(result, [{'type': 'int', 'value': 1}, {'type': 'int', 'value': 2}, {'type': 'int', 'value': 3}])

    def test_analyze_tensor(self):
        tensor = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
        result = self.api.analyze_tensor(tensor)
        self.assertEqual(result.get('type'), 'torch.Tensor')
        self.assertTrue(result.get('requires_grad'))
        self.assertTrue(os.path.exists(result.get('datapath')))

    def test_analyze_builtin(self):
        arg = slice(1, 10, 2)
        result = self.api.analyze_builtin(arg)
        self.assertEqual(result, {'type': 'slice', 'value': [1, 10, 2]})

    def test_transfer_types(self):
        data = 10
        dtype = 'int'
        result = self.api.transfer_types(data, dtype)
        self.assertEqual(result, 10)

    def test_is_builtin_class(self):
        element = 10
        result = self.api.is_builtin_class(element)
        self.assertEqual(result, True)

    def test_analyze_device_in_kwargs(self):
        element = torch.device('cuda:0')
        result = self.api.analyze_device_in_kwargs(element)
        self.assertEqual(result, {'type': 'torch.device', 'value': 'cuda:0'})

    def test_analyze_dtype_in_kwargs(self):
        element = torch.float32
        result = self.api.analyze_dtype_in_kwargs(element)
        self.assertEqual(result, {'type': 'torch.dtype', 'value': 'torch.float32'})

    def test_get_tensor_extremum(self):
        data = torch.tensor([1, 2, 3])
        result_max = self.api.get_tensor_extremum(data, 'max')
        result_min = self.api.get_tensor_extremum(data, 'min')
        self.assertEqual(result_max, 3)
        self.assertEqual(result_min, 1)

    def test_get_type_name(self):
        name = "<class 'int'>"
        result = self.api.get_type_name(name)
        self.assertEqual(result, 'int')

    def tearDown(self):
        if os.path.exists('./forward'):
            shutil.rmtree('./forward')