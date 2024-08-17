import sys
import logging
import os

import pytest
import mindspore
import torch
from unittest.mock import MagicMock

from msprobe.mindspore.api_accuracy_checker.base_compare_algorithm import compare_algorithms
from msprobe.mindspore.api_accuracy_checker.const import COSINE_SIMILARITY, MAX_ABSOLUTE_DIFF, MAX_RELATIVE_DIFF, ERROR

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)


@pytest.fixture
def mock_mstensor_compute_element():
    mock = MagicMock()
    mock.get_parameter.return_value = mindspore.Tensor([1., 1.9, 3.], dtype=mindspore.float32)
    mock.get_shape.return_value = (3,)
    return mock

@pytest.fixture
def mock_torchtensor_compute_element():
    mock = MagicMock()
    mock.get_parameter.return_value = torch.Tensor([1., 2., 3.], dtype=torch.float32)
    mock.get_shape.return_value = (3,)
    return mock


class TestClass:

    def test_cosine_similarity(self, mock_torchtensor_compute_element, mock_mstensor_compute_element):
        compare_result = compare_algorithms[COSINE_SIMILARITY](mock_torchtensor_compute_element, mock_mstensor_compute_element)
        assert abs(compare_result.compare_value - 0.9997375534689601) < 1e-5
        assert compare_result.pass_status == ERROR


    def test_max_absolute_difference(self, mock_torchtensor_compute_element, mock_mstensor_compute_element):
        compare_result = compare_algorithms[MAX_ABSOLUTE_DIFF](mock_torchtensor_compute_element, mock_mstensor_compute_element)
        assert abs(compare_result.compare_value - 0.1) < 1e-5
        assert compare_result.pass_status == ERROR

    def test_max_relative_difference(self, mock_torchtensor_compute_element, mock_mstensor_compute_element):
        compare_result = compare_algorithms[MAX_RELATIVE_DIFF](mock_torchtensor_compute_element, mock_mstensor_compute_element)
        assert abs(compare_result.compare_value - 0.05) < 1e-5
        assert compare_result.pass_status == ERROR