# coding=utf-8
import unittest
import pandas as pd
import multiprocessing
from msprobe.core.compare.multiprocessing_compute import _handle_multi_process, read_dump_data, check_accuracy
from msprobe.core.compare.acc_compare import Comparator
from msprobe.core.common.const import CompareConst
from msprobe.core.common.utils import CompareException


data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
         'torch.float32', 'torch.float32', [2, 2], [2, 2],
         '', '', '', '', '',
         1, 1, 1, 1, 1, 1, 1, 1,
         'Yes', '', '-1']]
o_data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
           'torch.float32', 'torch.float32', [2, 2], [2, 2],
           'None', 'None', 'None', 'None', 'None',
           1, 1, 1, 1, 1, 1, 1, 1,
           'None', 'No bench data matched', '-1']]
columns = CompareConst.COMPARE_RESULT_HEADER + ['Data_name']
result_df = pd.DataFrame(data, columns=columns)
o_result = pd.DataFrame(o_data, columns=columns)


class TestUtilsMethods(unittest.TestCase):

    def test_handle_multi_process(self):
        func = Comparator().compare_ops
        input_parma = {}
        lock = multiprocessing.Manager().RLock()
        result = _handle_multi_process(func, input_parma, result_df, lock)
        self.assertTrue(result.equals(o_result))

    def test_read_dump_data(self):
        result = read_dump_data(result_df)
        self.assertEqual(result, {'Functional.linear.0.forward.input.0': ['-1', '-1']})

        with self.assertRaises(CompareException) as context:
            result = read_dump_data(pd.DataFrame())
        self.assertEqual(context.exception.code, CompareException.INDEX_OUT_OF_BOUNDS_ERROR)

    def test_check_accuracy(self):
        max_abs_err = ''

        cos_1 = CompareConst.SHAPE_UNMATCH
        result_1 = check_accuracy(cos_1, max_abs_err)
        self.assertEqual(result_1, CompareConst.ACCURACY_CHECK_UNMATCH)

        cos_2 = CompareConst.NONE
        result_2 = check_accuracy(cos_2, max_abs_err)
        self.assertEqual(result_2, CompareConst.NONE)

        cos_3 = 'N/A'
        result_3 = check_accuracy(cos_3, max_abs_err)
        self.assertEqual(result_3, CompareConst.ACCURACY_CHECK_NO)

        cos_4 = ''
        result_4 = check_accuracy(cos_4, max_abs_err)
        self.assertEqual(result_4, CompareConst.NONE)

        cos_5 = 0.95
        max_abs_err = 0.002
        result_5 = check_accuracy(cos_5, max_abs_err)
        self.assertEqual(result_5, CompareConst.ACCURACY_CHECK_NO)

        cos_6 = 0.85
        max_abs_err = 2
        result_6 = check_accuracy(cos_6, max_abs_err)
        self.assertEqual(result_6, CompareConst.ACCURACY_CHECK_NO)

        cos_7 = 0.95
        max_abs_err = 0.001
        result_7 = check_accuracy(cos_7, max_abs_err)
        self.assertEqual(result_7, CompareConst.ACCURACY_CHECK_YES)
