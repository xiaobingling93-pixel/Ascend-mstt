import unittest
from typing import Any
from msprobe.core.overflow_check.utils import has_nan_inf


class TestHasNanInf(unittest.TestCase):
    def test_empty_dict(self):
        """Test with an empty dictionary"""
        self.assertFalse(has_nan_inf({}))

    def test_dict_without_nan_inf(self):
        """Test with a dictionary that doesn't contain NaN or Inf"""
        test_dict = {
            'Max': 10,
            'Min': 1,
            'Mean': 5,
            'Norm': 7
        }
        self.assertFalse(has_nan_inf(test_dict))

    def test_dict_with_nan(self):
        """Test dict with 'NaN' as a string in key values"""
        test_dict = {
            'Max': 'NaN',
            'Min': 5,
            'Mean': 3
        }
        self.assertTrue(has_nan_inf(test_dict))

    def test_dict_with_inf(self):
        """Test dict with 'Inf' as a string in key values"""
        test_dict = {
            'Max': 'Inf',
            'Min': 5,
            'Mean': 3
        }
        self.assertTrue(has_nan_inf(test_dict))

    def test_dict_with_lowercase_nan(self):
        """Test dict with lowercase 'nan'"""
        test_dict = {
            'Max': 'nan',
            'Min': 5,
            'Mean': 3
        }
        self.assertTrue(has_nan_inf(test_dict))

    def test_dict_with_lowercase_inf(self):
        """Test dict with lowercase 'inf'"""
        test_dict = {
            'Max': 'inf',
            'Min': 5,
            'Mean': 3
        }
        self.assertTrue(has_nan_inf(test_dict))

    def test_non_dict_input(self):
        """Test with non-dictionary input"""
        self.assertFalse(has_nan_inf(42))
        self.assertFalse(has_nan_inf("string"))
        self.assertFalse(has_nan_inf(None))
        self.assertFalse(has_nan_inf([1, 2, 3]))


def run_tests():
    """Run the tests"""
    unittest.main(exit=False)


if __name__ == '__main__':
    run_tests()
