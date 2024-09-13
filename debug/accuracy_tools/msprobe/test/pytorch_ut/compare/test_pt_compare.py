# coding=utf-8
import unittest
from msprobe.pytorch.compare.pt_compare import PTComparator
from msprobe.test.core_ut.compare.test_acc_compare import npu_dict, bench_dict

class TestUtilsMethods(unittest.TestCase):

    def test_check_op(self):
        fuzzy_match = False
        pt_comparator = PTComparator()
        result = pt_comparator.check_op(npu_dict, bench_dict, fuzzy_match)
        self.assertEqual(result, True)

    def test_match_op(self):
        fuzzy_match = False
        pt_comparator = PTComparator()
        a, b = pt_comparator.match_op([npu_dict], [bench_dict], fuzzy_match)
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

