import unittest

import numpy as np

from atat.pytorch.api_accuracy_checker.common.utils import CompareException
from atat.pytorch.api_accuracy_checker.compare.compare_utils import check_dtype_comparable, convert_str_to_float


class TestCompareUtils(unittest.TestCase):
    def test_check_dtype_comparable(self):
        x = np.array([1, 2, 3], dtype=np.int32)
        y = np.array([4, 5, 6], dtype=np.int32)
        self.assertTrue(check_dtype_comparable(x, y))

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        self.assertTrue(check_dtype_comparable(x, y))

        x = np.array([True, False, True], dtype=np.bool_)
        y = np.array([False, True, False], dtype=np.bool_)
        self.assertTrue(check_dtype_comparable(x, y))

        x = np.array([1, 2, 3], dtype=np.int32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        self.assertFalse(check_dtype_comparable(x, y))

        x = np.array([1, 2, 3], dtype=np.int32)
        y = np.array([True, False, True], dtype=np.bool_)
        self.assertFalse(check_dtype_comparable(x, y))

    def test_convert_str_to_float_when_valid_float(self):
        self.assertEqual(convert_str_to_float("123.45"), 123.45)

    def test_convert_str_to_float_when_valid_int(self):
        self.assertEqual(convert_str_to_float("123.0"), 123.0)

    def test_convert_str_to_float_when_valid_int_with_spaces(self):
        self.assertEqual(convert_str_to_float("   123.0   "), 123.0)

    def test_convert_str_to_float_when_empty_string(self):
        with self.assertRaises(CompareException) as cm:
            convert_str_to_float('')
        self.assertEqual(cm.exception.code, CompareException.INVALID_DATA_ERROR)
