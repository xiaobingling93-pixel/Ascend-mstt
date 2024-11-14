# coding=utf-8
import unittest
import numpy as np
from msprobe.core.compare.npy_compare import handle_inf_nan, get_error_type, reshape_value, get_error_message, \
    npy_data_check, statistics_data_check, GetCosineSimilarity, GetMaxAbsErr, get_relative_err, GetMaxRelativeErr, \
    GetThousandErrRatio, GetFiveThousandErrRatio, compare_ops_apply
from msprobe.core.common.const import CompareConst


op_name = 'Functional.conv2d.0.backward.input.0'


class TestUtilsMethods(unittest.TestCase):
    def test_handle_inf_nan_1(self):
        n_value = np.array([1, 2, np.inf, 4])
        b_value = np.array([1, 2, 3, 4])
        a, b = handle_inf_nan(n_value, b_value)
        self.assertTrue(a == CompareConst.NAN and b == CompareConst.NAN)

    def test_handle_inf_nan_2(self):
        n_value = np.array([1, 2, 3, 4])
        b_value = np.array([1, 2, np.nan, 4])
        a, b = handle_inf_nan(n_value, b_value)
        self.assertTrue(a == CompareConst.NAN and b == CompareConst.NAN)

    def test_handle_inf_nan_3(self):
        n_value = np.array([1, 2, 3, 4])
        b_value = np.array([1, 2, 3, 4])
        a, b = handle_inf_nan(n_value, b_value)
        self.assertTrue(np.array_equal(a, n_value) and np.array_equal(b, b_value))

    def test_get_error_type_1(self):
        n_value = np.array([1, 2, np.inf, 4])
        b_value = np.array([1, 2, 3, 4])
        error_flag = True
        a, b, c = get_error_type(n_value, b_value, error_flag)
        self.assertTrue(a == CompareConst.READ_NONE and b == CompareConst.READ_NONE and c == True)

    def test_get_error_type_2(self):
        n_value = np.array([1, 2, np.inf, 4])
        b_value = np.array([1, 2, 3, 4])
        error_flag = False
        a, b, c = get_error_type(n_value, b_value, error_flag)
        self.assertTrue(a == CompareConst.NAN and b == CompareConst.NAN and c == True)

    def test_get_error_type_3(self):
        n_value = np.array([1, 2, 3, 4])
        b_value = np.array([1, 2, 3, 4])
        error_flag = False
        a, b, c = get_error_type(n_value, b_value, error_flag)
        self.assertTrue(np.array_equal(a, n_value) and np.array_equal(b, b_value) and c == False)

    def test_get_error_type_4(self):
        n_value = np.array([])
        b_value = np.array([1, 2, 3, 4, 5])
        error_flag = False
        a, b, c = get_error_type(n_value, b_value, error_flag)
        self.assertTrue(a == CompareConst.NONE and b == CompareConst.NONE and c == True)

    def test_get_error_type_5(self):
        n_value = np.array([1, 2, 3, 4])
        b_value = np.array([1, 2, 3, 4, 5])
        error_flag = False
        a, b, c = get_error_type(n_value, b_value, error_flag)
        self.assertTrue(a == CompareConst.SHAPE_UNMATCH and b == CompareConst.SHAPE_UNMATCH and c == True)

    def test_reshape_value_1(self):
        n_value = np.array([[1, 2], [3, 4]])
        b_value = np.array([[1, 2, 3], [3, 4, 5]])
        a, b = reshape_value(n_value, b_value)
        self.assertTrue(np.array_equal(a, np.array([1., 2., 3., 4.])) and np.array_equal(b, np.array([1., 2., 3., 3., 4., 5.])))

    def test_reshape_value_2(self):
        n_value = np.array([])
        b_value = np.array([])
        a, b = reshape_value(n_value, b_value)
        self.assertTrue(np.array_equal(a, n_value) and np.array_equal(b, b_value))

    def test_get_error_message_True(self):
        b_value = CompareConst.READ_NONE
        error_flag = True

        n_value_1 = CompareConst.READ_NONE
        result_1 = get_error_message(n_value_1, b_value, op_name, error_flag, error_file='abc')
        self.assertEqual(result_1, 'Dump file: abc not found.')

        n_value_2 = CompareConst.READ_NONE
        result_2 = get_error_message(n_value_2, b_value, op_name, error_flag)
        self.assertEqual(result_2, CompareConst.NO_BENCH)

        n_value_3 = CompareConst.NONE
        result_3 = get_error_message(n_value_3, b_value, op_name, error_flag)
        self.assertEqual(result_3, 'This is empty data, can not compare.')

        n_value_4 = CompareConst.SHAPE_UNMATCH
        result_4 = get_error_message(n_value_4, b_value, op_name, error_flag)
        self.assertEqual(result_4, 'Shape of NPU and bench Tensor do not match. Skipped.')

        n_value_5 = CompareConst.NAN
        result_5 = get_error_message(n_value_5, b_value, op_name, error_flag)
        self.assertEqual(result_5, 'The position of inf or nan in NPU and bench Tensor do not match.')

    def test_get_error_message_False(self):
        b_value = CompareConst.READ_NONE
        error_flag = False

        n_value_1 = np.array(1)
        result_1 = get_error_message(n_value_1, b_value, op_name, error_flag, error_file='abc')
        self.assertEqual(result_1, 'This is type of scalar data, can not compare.')

        b_value = np.array([1])
        n_value_2 = np.array(['abc'])
        result_2 = get_error_message(n_value_2, b_value, op_name, error_flag)
        self.assertEqual(result_2, 'Dtype of NPU and bench Tensor do not match.')

    def test_data_check(self):
        n_value_1 = None
        b_value_1 = None
        error_flag_1, error_message_1 = npy_data_check(n_value_1, b_value_1)
        self.assertEqual(error_message_1, 'Dump file is not ndarray.\n')

        n_value_2 = ''
        b_value_2 = ''
        error_flag_2, error_message_2 = npy_data_check(n_value_2, b_value_2)
        self.assertEqual(error_message_2, 'Dump file is not ndarray.\n')

        n_value_3 = np.array([])
        b_value_3 = np.array([])
        error_flag_3, error_message_3 = npy_data_check(n_value_3, b_value_3)
        self.assertEqual(error_message_3, 'This is empty data, can not compare.\n')

        n_value_4 = np.array(1)
        b_value_4 = np.array(2)
        error_flag_4, error_message_4 = npy_data_check(n_value_4, b_value_4)
        self.assertEqual(error_message_4, 'This is type of scalar data, can not compare.\n')

        n_value_5 = np.array([1])
        b_value_5 = np.array([1, 2])
        error_flag_5, error_message_5 = npy_data_check(n_value_5, b_value_5)
        self.assertEqual(error_message_5, 'Shape of NPU and bench Tensor do not match.\n')

        n_value_6 = np.array([1, 2], dtype=float)
        b_value_6 = np.array([1, 2], dtype=int)
        error_flag_6, error_message_6 = npy_data_check(n_value_6, b_value_6)
        self.assertEqual(error_message_6, 'Dtype of NPU and bench Tensor do not match. Skipped.\n')

        n_value_7 = np.array([1, np.nan], dtype=float)
        b_value_7 = np.array([1, 2], dtype=float)
        error_flag_7, error_message_7 = npy_data_check(n_value_7, b_value_7)
        self.assertEqual(error_message_7, 'The position of inf or nan in NPU and bench Tensor do not match.\n')

        n_value_8 = np.array([1, 2], dtype=float)
        b_value_8 = np.array([1, 2], dtype=float)
        error_flag_8, error_message_8 = npy_data_check(n_value_8, b_value_8)
        self.assertFalse(error_flag_8)

    def test_statistics_data_check(self):
        result_dict_1 = {'NPU Name': None}
        error_flag_1, error_message_1 = statistics_data_check(result_dict_1)
        self.assertEqual(error_message_1, 'Dump file not found.\nThis is type of scalar data, can not compare.\n')
        self.assertTrue(error_flag_1)

        result_dict_2 = {'NPU Tensor Shape': [1], 'Bench Tensor Shape': [2]}
        error_flag_2, error_message_2 = statistics_data_check(result_dict_2)
        self.assertEqual(error_message_2, 'Dump file not found.\nTensor shapes do not match.\n')
        self.assertTrue(error_flag_2)

        result_dict_3 = {'NPU Dtype': 'torch.float32', 'Bench Dtype': 'torch.float16'}
        error_flag_3, error_message_3 = statistics_data_check(result_dict_3)
        self.assertEqual(error_message_3, 'Dump file not found.\nThis is type of scalar data, can not compare.\n''Dtype of NPU and bench Tensor do not match. Skipped.\n')
        self.assertTrue(error_flag_3)

    def test_GetCosineSimilarity_Ture(self):
        b_value = CompareConst.READ_NONE
        error_flag = True

        n_value_1 = CompareConst.READ_NONE
        op = GetCosineSimilarity()
        a_1, b_1 = op.apply(n_value_1, b_value, error_flag)
        self.assertEqual(a_1, CompareConst.UNSUPPORTED)
        self.assertEqual(b_1, '')

        n_value_2 = CompareConst.NONE
        a_2, b_2 = op.apply(n_value_2, b_value, error_flag)
        self.assertEqual(a_2, CompareConst.UNSUPPORTED)
        self.assertEqual(b_2, '')

        n_value_3 = CompareConst.SHAPE_UNMATCH
        a_3, b_3 = op.apply(n_value_3, b_value, error_flag)
        self.assertEqual(a_3, CompareConst.SHAPE_UNMATCH)
        self.assertEqual(b_3, '')

        n_value_4 = CompareConst.NAN
        a_4, b_4 = op.apply(n_value_4, b_value, error_flag)
        self.assertEqual(a_4, 'N/A')
        self.assertEqual(b_4, '')

    def test_GetCosineSimilarity_False(self):
        error_flag_2 = False
        b_value = CompareConst.READ_NONE

        n_value_5 = np.array(1)
        op = GetCosineSimilarity()
        a_5, b_5 = op.apply(n_value_5, b_value, error_flag_2)
        self.assertEqual(a_5, CompareConst.UNSUPPORTED)
        self.assertEqual(b_5, '')

        n_value_6 = np.array([1, 2])
        b_value_6 = np.array([1, 2])
        a_6, b_6 = op.apply(n_value_6, b_value_6, error_flag_2)
        self.assertEqual(a_6, 1.0)
        self.assertEqual(b_6, '')

        n_value_7 = np.array([0, 0])
        b_value_7 = np.array([0, 0])
        a_7, b_7 = op.apply(n_value_7, b_value_7, error_flag_2)
        self.assertEqual(a_7, 1.0)
        self.assertEqual(b_7, '')

        n_value_8 = np.array([0, 0])
        b_value_8 = np.array([1, 2])
        a_8, b_8 = op.apply(n_value_8, b_value_8, error_flag_2)
        self.assertEqual(a_8, CompareConst.NAN)
        self.assertEqual(b_8, 'Cannot compare by Cosine Similarity, All the data is Zero in npu dump data.')

        n_value_9 = np.array([1, 2])
        b_value_9 = np.array([0, 0])
        a_9, b_9 = op.apply(n_value_9, b_value_9, error_flag_2)
        self.assertEqual(a_9, CompareConst.NAN)
        self.assertEqual(b_9, 'Cannot compare by Cosine Similarity, All the data is Zero in Bench dump data.')

    def test_GetMaxAbsErr_True(self):
        b_value = CompareConst.READ_NONE
        error_flag = True

        n_value_1 = CompareConst.READ_NONE
        op = GetMaxAbsErr()
        a_1, b_1 = op.apply(n_value_1, b_value, error_flag)
        self.assertEqual(a_1, CompareConst.UNSUPPORTED)
        self.assertEqual(b_1, '')

        n_value_2 = CompareConst.NONE
        a_2, b_2 = op.apply(n_value_2, b_value, error_flag)
        self.assertEqual(a_2, 0)
        self.assertEqual(b_2, '')

        n_value_3 = CompareConst.SHAPE_UNMATCH
        a_3, b_3 = op.apply(n_value_3, b_value, error_flag)
        self.assertEqual(a_3, CompareConst.SHAPE_UNMATCH)
        self.assertEqual(b_3, '')

        n_value_4 = CompareConst.NAN
        a_4, b_4 = op.apply(n_value_4, b_value, error_flag)
        self.assertEqual(a_4, 'N/A')
        self.assertEqual(b_4, '')

    def test_GetMaxAbsErr_False(self):
        error_flag_2 = False

        n_value_5 = np.array([1, 2])
        b_value_5 = np.array([0, 0])
        op = GetMaxAbsErr()
        a_5, b_5 = op.apply(n_value_5, b_value_5, error_flag_2)
        self.assertEqual(a_5, 2.0)
        self.assertEqual(b_5, '')

    def test_get_relative_err(self):
        n_value = np.array([1, 2])
        b_value = np.array([1, 1])
        result = get_relative_err(n_value, b_value)
        self.assertTrue(np.array_equal(result, [0.0, 1.0]))

    def test_GetMaxRelativeErr_True(self):
        b_value = CompareConst.READ_NONE
        error_flag = True

        n_value_1 = CompareConst.READ_NONE
        op = GetMaxRelativeErr()
        a_1, b_1 = op.apply(n_value_1, b_value, error_flag)
        self.assertEqual(a_1, CompareConst.UNSUPPORTED)
        self.assertEqual(b_1, '')

        n_value_2 = CompareConst.NONE
        a_2, b_2 = op.apply(n_value_2, b_value, error_flag)
        self.assertEqual(a_2, 0)
        self.assertEqual(b_2, '')

        n_value_3 = CompareConst.SHAPE_UNMATCH
        a_3, b_3 = op.apply(n_value_3, b_value, error_flag)
        self.assertEqual(a_3, CompareConst.SHAPE_UNMATCH)
        self.assertEqual(b_3, '')

        n_value_4 = CompareConst.NAN
        a_4, b_4 = op.apply(n_value_4, b_value, error_flag)
        self.assertEqual(a_4, 'N/A')
        self.assertEqual(b_4, '')

    def test_GetMaxRelativeErr_False(self):
        error_flag_2 = False

        n_value_5 = np.array([1, 2])
        b_value_5 = np.array([1, 1])
        op = GetMaxRelativeErr()
        a_5, b_5 = op.apply(n_value_5, b_value_5, error_flag_2)
        self.assertEqual(a_5, 1.0)
        self.assertEqual(b_5, '')

    def test_GetThousandErrRatio_True(self):
        b_value = CompareConst.READ_NONE
        error_flag = True

        n_value_1 = CompareConst.READ_NONE
        op = GetThousandErrRatio()
        a_1, b_1 = op.apply(n_value_1, b_value, error_flag)
        self.assertEqual(a_1, CompareConst.UNSUPPORTED)
        self.assertEqual(b_1, '')

        n_value_2 = CompareConst.NONE
        a_2, b_2 = op.apply(n_value_2, b_value, error_flag)
        self.assertEqual(a_2, 0)
        self.assertEqual(b_2, '')

        n_value_3 = CompareConst.SHAPE_UNMATCH
        a_3, b_3 = op.apply(n_value_3, b_value, error_flag)
        self.assertEqual(a_3, CompareConst.SHAPE_UNMATCH)
        self.assertEqual(b_3, '')

        n_value_4 = CompareConst.NAN
        a_4, b_4 = op.apply(n_value_4, b_value, error_flag)
        self.assertEqual(a_4, 'N/A')
        self.assertEqual(b_4, '')

    def test_GetThousandErrRatio_False(self):
        error_flag_2 = False

        n_value_5 = np.array([1, 2])
        b_value_5 = np.array([1, 1])
        op = GetThousandErrRatio()
        a_5, b_5 = op.apply(n_value_5, b_value_5, error_flag_2)
        self.assertEqual(a_5, 0.5)
        self.assertEqual(b_5, '')

    def test_GetFiveThousandErrRatio_True(self):
        b_value = CompareConst.READ_NONE
        error_flag = True

        n_value_1 = CompareConst.READ_NONE
        op = GetFiveThousandErrRatio()
        a_1, b_1 = op.apply(n_value_1, b_value, error_flag)
        self.assertEqual(a_1, CompareConst.UNSUPPORTED)
        self.assertEqual(b_1, '')

        n_value_2 = CompareConst.NONE
        a_2, b_2 = op.apply(n_value_2, b_value, error_flag)
        self.assertEqual(a_2, 0)
        self.assertEqual(b_2, '')

        n_value_3 = CompareConst.SHAPE_UNMATCH
        a_3, b_3 = op.apply(n_value_3, b_value, error_flag)
        self.assertEqual(a_3, CompareConst.SHAPE_UNMATCH)
        self.assertEqual(b_3, '')

        n_value_4 = CompareConst.NAN
        a_4, b_4 = op.apply(n_value_4, b_value, error_flag)
        self.assertEqual(a_4, 'N/A')
        self.assertEqual(b_4, '')

    def test_GetFiveThousandErrRatio_False(self):
        error_flag_2 = False

        n_value_5 = np.array([1, 2])
        b_value_5 = np.array([1, 1])
        op = GetFiveThousandErrRatio()
        a_5, b_5 = op.apply(n_value_5, b_value_5, error_flag_2)
        self.assertEqual(a_5, 0.5)
        self.assertEqual(b_5, '')

    def test_compare_ops_apply(self):
        n_value = np.array([1, 1])
        b_value = np.array([1, 1])
        error_flag = False
        err_msg = ''
        a, b = compare_ops_apply(n_value, b_value, error_flag, err_msg)
        self.assertEqual(a, [1.0, 0.0, 0.0, 1.0, 1.0])
        self.assertEqual(b, '')