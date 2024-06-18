import unittest
import numpy as np
from api_accuracy_checker.compare.compare_utils import CompareConst, check_dtype_comparable

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
