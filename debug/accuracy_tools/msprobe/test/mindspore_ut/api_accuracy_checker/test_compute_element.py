import unittest
import sys
import logging
import os
import mindspore
import torch
import numpy as np

from msprobe.mindspore.api_accuracy_checker.compute_element import ComputeElement
from msprobe.mindspore.api_accuracy_checker.type_mapping import (FLOAT32, FLOAT_TYPE_STR, INT_TYPE_STR,
                                                                 TUPLE_TYPE_STR, STR_TYPE_STR, SLICE_TYPE_STR)
from msprobe.mindspore.api_accuracy_checker.utils import global_context
from msprobe.mindspore.api_accuracy_checker.const import MINDSPORE_PLATFORM, TORCH_PLATFORM

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)


class TestComputeElement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        class level setup_class
        """
        cls.init(TestComputeElement)

    def init(self):
        global_context.init(False, os.path.join(directory, "files"))
        self.ndarray = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        self.ms_tensor = mindspore.Tensor(self.ndarray)
        self.torch_tensor = torch.Tensor(self.ndarray)
        self.tensor_shape = (2, 3)


    def test_init_with_parameter_tensor(self):
        # input_parameter, origin_parameter, mstensor_parameter, torchtensor_parameter, shape, dtype_str
        parameter_results_mapping = [
            [self.ms_tensor, self.ms_tensor, self.ms_tensor, self.torch_tensor, self.tensor_shape, FLOAT32],
            [self.torch_tensor, self.torch_tensor, self.ms_tensor, self.torch_tensor, self.tensor_shape, FLOAT32],
        ]
        for parameter_result in parameter_results_mapping:
            input_parameter, origin_parameter, mstensor_parameter, torchtensor_parameter, shape, dtype_str = parameter_result

            compute_element = ComputeElement(parameter=input_parameter)

            self.assertTrue((compute_element.get_parameter(get_origin=True) == origin_parameter).all())
            self.assertTrue((compute_element.get_parameter(get_origin=False, tensor_platform=MINDSPORE_PLATFORM) == mstensor_parameter).all())
            self.assertTrue((compute_element.get_parameter(get_origin=False, tensor_platform=TORCH_PLATFORM) == torchtensor_parameter).all())
            self.assertEqual(compute_element.get_shape(), shape)
            self.assertEqual(compute_element.get_dtype(), dtype_str)

    def test_init_with_parameter_other_type(self):
        # input_parameter, origin_parameter, shape, dtype_str
        parameter_results_mapping = [
            [1, 1, tuple(), INT_TYPE_STR],
            [1.0, 1.0, tuple(), FLOAT_TYPE_STR],
            ["string", "string", tuple(), STR_TYPE_STR],
            [slice(1, 10, 2), slice(1, 10, 2), tuple(), SLICE_TYPE_STR],
            [tuple([1, 2]), tuple([1, 2]), tuple(), TUPLE_TYPE_STR],
        ]

        for parameter_result in parameter_results_mapping:
            input_parameter, origin_parameter, shape, dtype_str = parameter_result

            compute_element = ComputeElement(parameter=input_parameter)

            self.assertEqual(compute_element.get_parameter(get_origin=True), origin_parameter)
            self.assertEqual(compute_element.get_shape(), shape)
            self.assertEqual(compute_element.get_dtype(), dtype_str)

    def test_init_with_compute_element_info_mstensor(self):
        global_context.is_constructed = False
        compute_element_info = {
            "type": "mindspore.Tensor",
            "dtype": "Float32",
            "shape":[2, 3],
            "Max": 3.0,
            "Min": 1.0,
            "data_name": "input.npy"
        }
        compute_element = ComputeElement(compute_element_info=compute_element_info)
        self.assertTrue((compute_element.get_parameter(get_origin=True) == self.ms_tensor).all())
        self.assertEqual(compute_element.get_shape(), self.tensor_shape)
        self.assertEqual(compute_element.get_dtype(), FLOAT32)

    def test_init_with_compute_element_info_mstensor_constructed(self):
        global_context.is_constructed = True
        compute_element_info = {
            "type": "mindspore.Tensor",
            "dtype": "Float32",
            "shape":[2, 3],
            "Max": 3.0,
            "Min": 1.0,
            "data_name": "input.npy"
        }
        compute_element = ComputeElement(compute_element_info=compute_element_info)
        parameter = compute_element.get_parameter(get_origin=True)
        self.assertTrue((parameter <= 3.0).all())
        self.assertTrue((parameter >= 1.0).all())
        self.assertEqual(compute_element.get_shape(), self.tensor_shape)
        self.assertEqual(compute_element.get_dtype(), FLOAT32)

    def test_init_with_compute_element_info_tuple(self):
        global_context.is_constructed = False
        compute_element_info = [
            {
                "type": "mindspore.Tensor",
                "dtype": "Float32",
                "shape":[2, 3],
                "Max": 3.0,
                "Min": 1.0,
                "data_name": "input.npy"
            },
            {
                "type": "mindspore.Tensor",
                "dtype": "Float32",
                "shape":[2, 3],
                "Max": 3.0,
                "Min": 1.0,
                "data_name": "input.npy"
            },
        ]
        compute_element = ComputeElement(compute_element_info=compute_element_info)
        parameter = compute_element.get_parameter(get_origin=True)
        self.assertTrue((parameter[0] == self.ms_tensor).all())
        self.assertTrue((parameter[1] == self.ms_tensor).all())
        self.assertEqual(compute_element.get_shape(), tuple())
        self.assertEqual(compute_element.get_dtype(), TUPLE_TYPE_STR)

    def test_init_with_compute_element_info_int(self):
        compute_element_info = {
            "type": "int",
            "value": -1,
        }
        compute_element = ComputeElement(compute_element_info=compute_element_info)
        parameter = compute_element.get_parameter(get_origin=True)
        self.assertEqual(parameter, -1)
        self.assertEqual(compute_element.get_shape(), tuple())
        self.assertEqual(compute_element.get_dtype(), INT_TYPE_STR)

if __name__ == '__main__':
    unittest.main()
