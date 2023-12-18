import unittest
import torch
import numpy as np
from ptdbg_ascend.dump.dump import *

class TestDump(unittest.TestCase):

    def setUp(self):
        self.tensor = torch.tensor([1.0, 2.0, 3.0])
        self.scalar = 5.0
        self.prefix = "test_prefix"
        self.dump_step = 1
        self.dump_file_name = "test_file"

    def test_get_not_float_tensor_info(self):
        data_info = get_not_float_tensor_info(self.tensor)
        self.assertEqual(data_info.save_data.tolist(), self.tensor.numpy().tolist())
        self.assertEqual(data_info.summary_data, [3.0, 1.0, 2.0, 'Nan'])
        self.assertEqual(data_info.dtype, 'torch.float32')
        self.assertEqual(data_info.shape, (3,))

    def test_get_scalar_data_info(self):
        data_info = get_scalar_data_info(self.scalar)
        self.assertEqual(data_info.save_data, self.scalar)
        self.assertEqual(data_info.summary_data, [self.scalar, self.scalar, self.scalar, self.scalar])
        self.assertEqual(data_info.dtype, '<class \'float\'>')
        self.assertEqual(data_info.shape, '[]')

    def test_get_float_tensor_info(self):
        data_info = get_float_tensor_info(self.tensor)
        self.assertEqual(data_info.save_data.tolist(), self.tensor.numpy().tolist())
        self.assertEqual(data_info.summary_data, [3.0, 1.0, 2.0, 3.7416574954986572])
        self.assertEqual(data_info.dtype, 'torch.float32')
        self.assertEqual(data_info.shape, (3,))

    def test_get_tensor_data_info(self):
        tensor_max = 3.0
        tensor_min = 1.0
        tensor_mean = 2.0
        data_info = get_tensor_data_info(self.tensor, tensor_max, tensor_min, tensor_mean)
        self.assertEqual(data_info.save_data.tolist(), self.tensor.numpy().tolist())
        self.assertEqual(data_info.summary_data, [tensor_max, tensor_min, tensor_mean])
        self.assertEqual(data_info.dtype, 'torch.float32')
        self.assertEqual(data_info.shape, (3,))

    def test_json_dump_condition(self):
        result = json_dump_condition(self.prefix)
        self.assertEqual(result, False)

    def test_get_pkl_file_path(self):
        result = get_pkl_file_path()
        self.assertEqual(result, "")

