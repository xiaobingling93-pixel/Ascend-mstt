import unittest
import numpy as np
import torch
from api_accuracy_checker.compare import compare as cmp
from api_accuracy_checker.compare import algorithm as alg

class TestAlgorithmMethods(unittest.TestCase):

    def test_get_max_abs_err(self):
        b_value = np.array([1.0, 2.0, 3.0])
        n_value = np.array([1.0, 2.0, 3.0])
        abs_err = np.abs(b_value - n_value)
        self.assertEqual(alg.get_max_abs_err(abs_err), (0.0, True))

    def test_get_rel_err_ratio_thousandth(self):
        b_value = np.array([1.0, 2.0, 3.0])
        n_value = np.array([1.0, 2.0, 3.0])
        abs_err = np.abs(b_value - n_value)
        rel_err = alg.get_rel_err_origin(abs_err, b_value)
        self.assertEqual(alg.get_rel_err_ratio(rel_err, 0.001), (1.0, True))

    def test_get_rel_err_ratio_ten_thousandth(self):
        b_value = np.array([1.0, 2.0, 3.0])
        n_value = np.array([1.0, 2.0, 3.0])
        abs_err = np.abs(b_value - n_value)
        rel_err = alg.get_rel_err_origin(abs_err, b_value)
        self.assertEqual(alg.get_rel_err_ratio(rel_err, 0.0001), (1.0, True))

    def test_cosine_sim(self):
        cpu_output = np.array([1.0, 2.0, 3.0])
        npu_output = np.array([1.0, 2.0, 3.0])
        self.assertEqual(alg.cosine_sim(cpu_output, npu_output), (1.0, True, ''))
