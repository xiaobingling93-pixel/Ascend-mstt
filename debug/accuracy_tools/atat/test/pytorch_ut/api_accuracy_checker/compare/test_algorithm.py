import unittest
import numpy as np
from atat.pytorch.api_accuracy_checker.compare import algorithm as alg


class TestAlgorithmMethods(unittest.TestCase):

    def setUp(self):
        self.bench_data = np.array([1.0, 1.0, 9.0], dtype=np.float16)
        self.device_data = np.array([5.0, 2.0, 1.0], dtype=np.float16)
        self.abs_err = np.abs(self.device_data - self.bench_data)
        self.rel_err_orrign = np.abs(self.abs_err / self.bench_data)
        eps = np.finfo(self.bench_data.dtype).eps
        self.abs_bench = np.abs(self.bench_data)
        self.abs_bench_with_eps = self.abs_bench + eps
        self.rel_err = self.abs_err / self.abs_bench_with_eps

    def test_cosine_sim(self):
        cpu_output = np.array([1.0, 2.0, 3.0])
        npu_output = np.array([1.0, 2.0, 3.0])
        self.assertEqual(alg.cosine_sim(cpu_output, npu_output), (1.0, True, ''))

    def test_get_rmse(self):
        inf_nan_mask = [False, False, False]
        self.assertAlmostEqual(alg.get_rmse(self.abs_err, inf_nan_mask), 5.196, 3)

    def test_get_error_balance(self):
        self.assertEqual(alg.get_error_balance(self.bench_data, self.device_data), 1 / 3)

    def test_get_small_value_err_ratio(self):
        small_value_mask = [True, True, True, False, True]
        abs_err_greater_mask = [False, True, True, True, False]
        self.assertEqual(alg.get_small_value_err_ratio(small_value_mask, abs_err_greater_mask), 0.5)

    def get_rel_err(self):
        eps = np.finfo(self.bench_data.dtype).eps
        abs_bench = np.abs(self.bench_data)
        abs_bench_with_eps = abs_bench + eps
        small_value_mask = [False, False, False]
        inf_nan_mask = [False, False, False]
        rel_err = self.abs_err / abs_bench_with_eps
        self.assertListEqual(list(alg.get_rel_err(self.abs_err, abs_bench_with_eps, small_value_mask, inf_nan_mask)),
                             list(rel_err))

    def test_get_abs_err(self):
        self.assertListEqual(list(alg.get_abs_err(self.bench_data, self.device_data)), [4.0, 1.0, 8.0])

    def test_get_rel_err_origin(self):
        self.assertListEqual(list(alg.get_rel_err_origin(self.abs_err, self.bench_data)), list(self.rel_err_orrign))

    def test_get_max_abs_err(self):
        self.assertEqual(alg.get_max_abs_err(self.abs_err), (8.0, False))

    def test_get_max_rel_err(self):
        self.assertAlmostEqual(alg.get_max_rel_err(self.rel_err), 3.996, 3)

    def test_get_mean_rel_err(self):
        self.assertAlmostEqual(alg.get_mean_rel_err(self.rel_err), 1.961, 3)

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

    def test_get_finite_and_infinite_mask(self):
        both_finite_mask, inf_nan_mask = alg.get_finite_and_infinite_mask(self.bench_data, self.device_data)
        self.assertListEqual(list(both_finite_mask), [True, True, True])
        self.assertListEqual(list(inf_nan_mask), [False, False, False])

    def test_get_small_value_mask(self):
        b_value = np.array([1e-7, 1.0, 2e-6], dtype=np.float16)
        abs_bench = np.abs(b_value)
        both_finite_mask = [True, True, True]
        small_value_mask = alg.get_small_value_mask(abs_bench, both_finite_mask, 1e-3)
        self.assertListEqual(list(small_value_mask), [True, False, True])

    def test_get_abs_bench_with_eps(self):
        abs_bench, abs_bench_with_eps = alg.get_abs_bench_with_eps(self.bench_data, np.float16)
        self.assertListEqual(list(abs_bench), list(self.abs_bench))
        self.assertListEqual(list(abs_bench_with_eps), list(self.abs_bench_with_eps))

    def test_check_inf_nan_value(self):
        both_finite_mask, inf_nan_mask = alg.get_finite_and_infinite_mask(self.bench_data, self.device_data)
        self.assertEqual(alg.check_inf_nan_value(inf_nan_mask, self.bench_data, self.device_data, np.float16, 0.001), 0)

    def test_check_small_value(self):
        a_value = np.array([1e-7, 1.0, 2e-6], dtype=np.float16)
        b_value = np.array([1e-7, 1.0, 2e-6], dtype=np.float16)
        abs_bench = np.abs(b_value)
        both_finite_mask = [True, True, True]
        abs_err = abs(a_value - b_value)
        small_value_mask = alg.get_small_value_mask(abs_bench, both_finite_mask, 1e-3)
        self.assertEqual(alg.check_small_value(abs_err, small_value_mask, 0.001), 0)

    def test_check_norm_value(self):
        both_finite_mask, inf_nan_mask = alg.get_finite_and_infinite_mask(self.bench_data, self.device_data)
        small_value_mask = alg.get_small_value_mask(self.abs_bench, both_finite_mask, 1e-3)
        normal_value_mask = np.logical_and(both_finite_mask, np.logical_not(small_value_mask))
        print(normal_value_mask)
        print(self.rel_err)
        self.assertEqual(alg.check_norm_value(normal_value_mask, self.rel_err, 0.001), 1)
