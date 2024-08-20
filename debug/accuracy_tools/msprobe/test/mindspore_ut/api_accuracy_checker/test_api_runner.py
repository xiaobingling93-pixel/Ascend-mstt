import sys
import logging
import os

import pytest
import mindspore
import torch
from unittest.mock import MagicMock

from msprobe.mindspore.api_accuracy_checker.api_runner import api_runner, ApiInputAggregation
from msprobe.mindspore.api_accuracy_checker.const import MINDSPORE_PLATFORM, TORCH_PLATFORM, FORWARD_API, BACKWARD_API

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)


# 创建一个包含if判断的mock实例的fixture
@pytest.fixture
def mock_compute_element_input_instance():
    mock = MagicMock()
    def side_effect(**kwargs):
        if kwargs.get("tensor_platform") == MINDSPORE_PLATFORM:
            return mindspore.Tensor([1., 2., 3.])
        else:
            return torch.Tensor([1., 2., 3.])
    mock.get_parameter.side_effect = side_effect
    return mock

@pytest.fixture
def mock_compute_element_kwargs_instance():
    mock = MagicMock()
    mock.get_parameter.return_value = "tanh"
    return mock

@pytest.fixture
def mock_compute_element_forward_result_instance():
    mock = MagicMock()
    def side_effect(**kwargs):
        if kwargs.get("tensor_platform") == MINDSPORE_PLATFORM:
            return mindspore.Tensor([8.41192007e-01, 1.95459759e+00, 2.99636269e+00])
        else:
            return torch.Tensor([8.41192007e-01, 1.95459759e+00, 2.99636269e+00])
    mock.get_parameter.side_effect = side_effect
    return mock

@pytest.fixture
def mock_compute_element_backward_result_instance():
    mock = MagicMock()
    def side_effect(**kwargs):
        if kwargs.get("tensor_platform") == MINDSPORE_PLATFORM:
            return mindspore.Tensor([1.0833155, 2.1704636, 3.0358372])
        else:
            return torch.Tensor([1.0833155, 2.1704636, 3.0358372])
    mock.get_parameter.side_effect = side_effect
    return mock

class TestClass:

    def test_run_api(self, mock_compute_element_kwargs_instance, mock_compute_element_input_instance,
                     mock_compute_element_forward_result_instance, mock_compute_element_backward_result_instance):
        kwargs = {"approximate": mock_compute_element_kwargs_instance}
        inputs = [mock_compute_element_input_instance]
        gradient_inputs = [mock_compute_element_input_instance]
        forward_result = [mock_compute_element_forward_result_instance]
        backward_result = [mock_compute_element_backward_result_instance]

        forward_api_input_aggregation = ApiInputAggregation(inputs, kwargs, None)
        backward_api_input_aggregation = ApiInputAggregation(inputs, {}, gradient_inputs)


        # api_instance, api_input_aggregation, forward_or_backward, api_platform, result
        test_cases = [
            [mindspore.mint.nn.functional.gelu, forward_api_input_aggregation, FORWARD_API, MINDSPORE_PLATFORM, forward_result],
            [mindspore.mint.nn.functional.gelu, backward_api_input_aggregation, BACKWARD_API, MINDSPORE_PLATFORM, backward_result],
            [torch.nn.functional.gelu, forward_api_input_aggregation, FORWARD_API, TORCH_PLATFORM, forward_result],
            [torch.nn.functional.gelu, backward_api_input_aggregation, BACKWARD_API, TORCH_PLATFORM, backward_result],
        ]
        for test_case in test_cases:
            api_instance, api_input_aggregation, forward_or_backward, api_platform, results_target = test_case
            results_real = api_runner.run_api(api_instance, api_input_aggregation, forward_or_backward, api_platform)
            for res_real, res_target in zip(results_real, results_target):
                assert (abs(res_real.get_parameter() - res_target.get_parameter(tensor_platform=api_platform)) < 1e-5).all()


    def test_get_api_instance(self):
        #api_type_str, api_sub_name, api_platform, result_api
        test_cases = [
            ["MintFunctional", "relu", MINDSPORE_PLATFORM, mindspore.mint.nn.functional.relu],
            ["MintFunctional", "relu", TORCH_PLATFORM, torch.nn.functional.relu]
        ]
        for test_case in test_cases:
            api_type_str, api_sub_name, api_platform, result_api = test_case
            assert api_runner.get_api_instance(api_type_str, api_sub_name, api_platform) == result_api

    def test_get_info_from_name(self):
        api_name = "MintFunctional.relu.0.backward"
        api_type_str, api_sub_name = api_runner.get_info_from_name(api_name_str=api_name)
        assert api_type_str == "MintFunctional"
        assert api_sub_name == "relu"
