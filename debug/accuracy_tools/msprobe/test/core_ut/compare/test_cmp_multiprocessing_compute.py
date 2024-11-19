# coding=utf-8
import unittest
import threading
import pandas as pd
import multiprocessing
from msprobe.core.compare.multiprocessing_compute import _handle_multi_process, read_dump_data, ComparisonResult, \
    _save_cmp_result, check_accuracy
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
           'unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported',
           1, 1, 1, 1, 1, 1, 1, 1,
           'None', 'No bench data matched.', '-1']]
columns = CompareConst.COMPARE_RESULT_HEADER + ['Data_name']
result_df = pd.DataFrame(data, columns=columns)
o_result = pd.DataFrame(o_data, columns=columns)


class TestUtilsMethods(unittest.TestCase):

    def setUp(self):
        self.result_df = pd.DataFrame(columns=[
            CompareConst.COSINE, CompareConst.MAX_ABS_ERR, CompareConst.MAX_RELATIVE_ERR,
            CompareConst.ERROR_MESSAGE, CompareConst.ACCURACY,
            CompareConst.ONE_THOUSANDTH_ERR_RATIO, CompareConst.FIVE_THOUSANDTHS_ERR_RATIO
        ])
        self.lock = threading.Lock()

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

    def test_save_cmp_result_success(self):
        comparison_result = ComparisonResult(
            cos_result=[0.99, 0.98],
            max_err_result=[0.01, 0.02],
            max_relative_err_result=[0.001, 0.002],
            err_msgs=['', 'Error in comparison'],
            one_thousand_err_ratio_result=[0.1, 0.2],
            five_thousand_err_ratio_result=[0.05, 0.1]
        )
        offset = 0
        updated_df = _save_cmp_result(offset, comparison_result, self.result_df, self.lock)

        self.assertEqual(updated_df.loc[0, CompareConst.COSINE], 0.99)
        self.assertEqual(updated_df.loc[1, CompareConst.COSINE], 0.98)
        self.assertEqual(updated_df.loc[1, CompareConst.ERROR_MESSAGE], 'Error in comparison')

    def test_save_cmp_result_index_error(self):
        comparison_result = ComparisonResult(
            cos_result=[0.99],
            max_err_result=[],
            max_relative_err_result=[0.001],
            err_msgs=[''],
            one_thousand_err_ratio_result=[0.1],
            five_thousand_err_ratio_result=[0.05]
        )
        with self.assertRaises(CompareException) as context:
            _save_cmp_result(0, comparison_result, self.result_df, self.lock)
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
