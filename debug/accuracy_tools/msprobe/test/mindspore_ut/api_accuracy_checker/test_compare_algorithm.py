import unittest
import sys
import logging
import os
import mindspore
import torch
from unittest.mock import MagicMock

from msprobe.mindspore.api_accuracy_checker.base_compare_algorithm import compare_algorithms
from msprobe.mindspore.api_accuracy_checker.const import COSINE_SIMILARITY, MAX_ABSOLUTE_DIFF, MAX_RELATIVE_DIFF, ERROR

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)


class TestCompareAlgorithms(unittest.TestCase):

    def setUp(self):
        self.mock_torchtensor_compute_element = MagicMock()
        self.mock_torchtensor_compute_element.get_parameter.return_value = torch.Tensor([1., 2., 3.])
        self.mock_torchtensor_compute_element.get_shape.return_value = (3,)
        self.mock_mstensor_compute_element = MagicMock()
        self.mock_mstensor_compute_element.get_parameter.return_value = mindspore.Tensor([1., 1.9, 3.])
        self.mock_mstensor_compute_element.get_shape.return_value = (3,)

    def test_cosine_similarity(self):
        compare_result = compare_algorithms[COSINE_SIMILARITY](self.mock_torchtensor_compute_element, self.mock_mstensor_compute_element)
        self.assertAlmostEqual(compare_result.compare_value, 0.9997375534689601, places=5)
        self.assertEqual(compare_result.pass_status, ERROR)

    def test_max_absolute_difference(self):
        compare_result = compare_algorithms[MAX_ABSOLUTE_DIFF](self.mock_torchtensor_compute_element, self.mock_mstensor_compute_element)
        self.assertAlmostEqual(compare_result.compare_value, 0.1, places=5)
        self.assertEqual(compare_result.pass_status, ERROR)

    def test_max_relative_difference(self):
        compare_result = compare_algorithms[MAX_RELATIVE_DIFF](self.mock_torchtensor_compute_element, self.mock_mstensor_compute_element)
        self.assertAlmostEqual(compare_result.compare_value, 0.05, places=5)
        self.assertEqual(compare_result.pass_status, ERROR)

if __name__ == '__main__':
    unittest.main()
