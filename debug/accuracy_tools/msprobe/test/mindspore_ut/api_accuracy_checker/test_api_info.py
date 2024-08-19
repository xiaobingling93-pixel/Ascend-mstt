import sys
import logging
import os

import pytest
import mindspore

from msprobe.mindspore.api_accuracy_checker.api_info import ApiInfo
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

    def test_get_compute_element_list(self):
        # first load forward backward api_info
        forward_api_info_dict = {
            "input_args": [
                {
                    "type": "mindspore.Tensor",
                    "dtype": "Float32",
                    "shape": [
                        2,
                        3
                    ],
                    "Max": 3.0,
                    "Min": 1.0,
                    "data_name": "input.npy",
                }
            ],
            "input_kwargs": {
                "approximate": {
                    "type": "str",
                    "value": "tanh",
                }
            },
            "output": [
                                {
                    "type": "mindspore.Tensor",
                    "dtype": "Float32",
                    "shape": [
                        2,
                        3
                    ],
                    "Max": 3.0,
                    "Min": 1.0,
                    "data_name": "input.npy",
                }
            ],
        }

        api_info = ApiInfo("MintFuntional.gelu.0.forward")
        api_info.load_forward_info(forward_api_info_dict)

        assert api_info.check_forward_info == True
        assert api_info.check_backward_info == False

        input_compute_element_list = api_info.get_compute_element_list("forward_api", "input")
        parameter_real = input_compute_element_list[0].get_parameter()
        parameter_target = mindspore.Tensor([1., 2., 3.])
        assert parameter_real == parameter_target

        kwargs_compute_element_dict = api_info.get_kwargs()
        assert kwargs_compute_element_dict.get("approximate").get_parameter() == "tanh"

