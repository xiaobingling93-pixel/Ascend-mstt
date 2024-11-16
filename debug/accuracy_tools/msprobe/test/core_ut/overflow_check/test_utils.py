import unittest

import numpy as np

from msprobe.core.overflow_check.utils import has_nan_inf


class TestUtils(unittest.TestCase):
    def test_has_nan_inf_with_inf_nan(self):
        test_cases = [
            {"Max": "inf", "Min": "1.0"},
            {"Mean": "1.0", "Norm": "nan"},
            {"Max": "INF", "Min": "1.0"},
            {"Norm": float('inf'), "Mean": "1.0"},
            {"Max": "-inf", "Min": "1.0"},
            {"Mean": "-Inf", "Norm": "2.0"},
            {"Norm": float('-inf'), "Mean": "1.0"}
        ]

        for data in test_cases:
            with self.subTest(data=data):
                self.assertTrue(has_nan_inf(data),
                                f"Failed to detect Inf, nan in {data}")

    def test_has_nan_inf_normal_values(self):
        """Test normal numerical values"""
        test_cases = [
            {"Max": "1.0", "Min": "0.0", "Mean": "0.5", "Norm": "2.0"},
            {"Max": 1.0, "Min": 0.0, "Mean": 0.5, "Norm": 2.0},
            {"Max": "1e-10", "Min": "-1e10", "Mean": "0", "Norm": "1"},
            {"Max": str(np.float32(1.0)), "Min": str(np.float64(0.0))}
        ]

        for data in test_cases:
            with self.subTest(data=data):
                self.assertFalse(has_nan_inf(data),
                                 f"Incorrectly detected NaN/Inf in {data}")

    def test_has_nan_inf_edge_cases(self):
        test_cases = [
            {},  # Empty
            {"other_key": "nan"},  # nan
            {"other_key": "inf"},  # inf
            {"Max": None},  # None
            {"Mean": ""},  # Empty
            {"Norm": "undefined"},  # Undefined string
            None,  # None
            "not a dict",  # String
            123,  # Number
            [],  # List
            {"Max": "1.0e+308"},
            {"Min": "1.0e-308"}
        ]

        for data in test_cases:
            with self.subTest(data=data):
                self.assertFalse(has_nan_inf(data),
                                 f"Incorrectly detected NaN/Inf in {data}")


if __name__ == '__main__':
    unittest.main()
