import sys
import logging
import os

import pytest
import mindspore
import torch
import numpy as np

from msprobe.mindspore.api_accuracy_checker.compute_element import ComputeElement
from msprobe.mindspore.api_accuracy_checker.type_mapping import FLOAT32, FLOAT_TYPE_STR
from msprobe.mindspore.api_accuracy_checker.utils import global_context

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)



class TestClass:
    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    def init(self):
        global_context.init(False, os.path.join(directory, "files"))
        self.ndarray = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        self.ms_tensor = mindspore.Tensor(self.ndarray)
        self.torch_tensor = torch.Tensor(self.ndarray)
        self.tensor_shape = (2, 3)
        self.float_instance = 12.0

        pass

    def test_init_with_parameter_mstensor(self):
        # input_parameter, origin_parameter, mstensor_parameter, torchtensor_parameter, shape, dtype_str
        parameter_results_mapping = [
            [self.ms_tensor, self.ms_tensor, self.ms_tensor, self.torch_tensor, self.tensor_shape, FLOAT32],
            [self.torch_tensor, self.torch_tensor, self.ms_tensor, self.torch_tensor, self.tensor_shape, FLOAT32],
            [self.float_instance, self.float_instance, self.float_instance, self.float_instance, tuple(), FLOAT_TYPE_STR],

        ]
        for parameter_result in parameter_results_mapping:
            input_parameter, origin_parameter, mstensor_parameter, torchtensor_parameter, shape, dtype_str = parameter_result

            compute_element = ComputeElement(parameter=input_parameter)

            assert compute_element.get_parameter(get_origin=True) == origin_parameter
            assert compute_element.get_parameter(get_origin=False, get_mindspore_tensor=True) == mstensor_parameter
            assert compute_element.get_parameter(get_origin=False, get_mindspore_tensor=False) == torchtensor_parameter
            assert compute_element.get_shape() == shape
            assert compute_element.get_dtype() == dtype_str

    def test_init_with_compute_element_info(self):
        compute_element_info = {
            "type": "mindspore.Tensor",
            "dtype": "Float32",
            "shape":[2, 3],
            "Max": 3.0,
            "Min": 1.0,
            "data_name": "input.npy"
        }
        compute_element = ComputeElement(compute_element_info=compute_element_info)
        assert compute_element.get_parameter(get_origin=True) == self.ms_tensor
        assert compute_element.get_shape() == self.tensor_shape
        assert compute_element.get_dtype() == FLOAT32



