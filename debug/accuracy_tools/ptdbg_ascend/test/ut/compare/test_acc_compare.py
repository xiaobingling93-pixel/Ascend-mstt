# coding=utf-8
import unittest
import numpy as np
import os
from ptdbg_ascend.compare import acc_compare as compare
from ptdbg_ascend.common.utils import CompareConst


npu_dict = {'op_name': ['Functional_conv2d_0_forward_input.0', 'Functional_conv2d_0_forward_input.1', 'Functional_conv2d_0_forward_input.2', 'Functional_conv2d_0_forward_output'],\
 'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]), ('torch.float32', [16])],\
  'output_struct': [('torch.float32', [1, 16, 28, 28])], 'summery': [[3.029174327850342, -2.926689624786377, -0.06619918346405029], \
  [0.19919930398464203, -0.19974489510059357, 0.006269412115216255], [0.19734230637550354, -0.18177609145641327, 0.007903944700956345], [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}
bench_dict = {'op_name': ['Functional_conv2d_0_forward_input.0', 'Functional_conv2d_0_forward_input.1', 'Functional_conv2d_0_forward_input.2', 'Functional_conv2d_0_forward_output'],\
 'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]), ('torch.float32', [16])],\
  'output_struct': [('torch.float32', [1, 16, 28, 28])], 'summery': [[3.029174327850342, -2.926689624786377, -0.06619918346405029], \
  [0.19919930398464203, -0.19974489510059357, 0.006269412115216255], [0.19734230637550354, -0.18177609145641327, 0.007903944700956345], [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}
tensor_list = [['Functional_conv2d_0_forward_input.0', 1, [], 'torch.float32', [1, 1, 28, 28], [3.029174327850342, -2.926689624786377, -0.06619918346405029]],\
 ['Functional_conv2d_0_forward_input.1', 1, [], 'torch.float32', [16, 1, 5, 5], [0.19919930398464203, -0.19974489510059357, 0.006269412115216255]], \
 ['Functional_conv2d_0_forward_input.2', 1, [], 'torch.float32', [16], [0.19734230637550354, -0.18177609145641327, 0.007903944700956345]],\
  ['Functional_conv2d_0_forward_output', 1, [], 'torch.float32', [1, 16, 28, 28], [2.1166646480560303, -2.190781354904175, -0.003579073818400502]]]
result_op_dict = {'op_name': ['Functional_conv2d_0_forward_input.0', 'Functional_conv2d_0_forward_input.1', 'Functional_conv2d_0_forward_input.2', 'Functional_conv2d_0_forward_output'], \
'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]), ('torch.float32', [16])], \
'output_struct': [('torch.float32', [1, 16, 28, 28])], 'summery': [[3.029174327850342, -2.926689624786377, -0.06619918346405029], [0.19919930398464203, -0.19974489510059357, 0.006269412115216255], \
[0.19734230637550354, -0.18177609145641327, 0.007903944700956345], [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

o_result = [['Functional_conv2d_0_forward_input.0', 'Functional_conv2d_0_forward_input.0', 'torch.float32', 'torch.float32', [1, 1, 28, 28], [1, 1, 28, 28], ' ', ' ', ' ', ' ', ' ', 3.029174327850342, -2.926689624786377, -0.06619918346405029, 3.029174327850342, -2.926689624786377, -0.06619918346405029, 'Yes', ''], ['Functional_conv2d_0_forward_input.1', 'Functional_conv2d_0_forward_input.1', 'torch.float32', 'torch.float32', [16, 1, 5, 5], [16, 1, 5, 5], ' ', ' ', ' ', ' ', ' ', 0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 'Yes', ''], ['Functional_conv2d_0_forward_input.2', 'Functional_conv2d_0_forward_input.2', 'torch.float32', 'torch.float32', [16], [16], ' ', ' ', ' ', ' ', ' ', 0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 'Yes', ''], ['Functional_conv2d_0_forward_output', 'Functional_conv2d_0_forward_output', 'torch.float32', 'torch.float32', [1, 16, 28, 28], [1, 16, 28, 28], ' ', ' ', ' ', ' ', ' ', 2.1166646480560303, -2.190781354904175, -0.003579073818400502, 2.1166646480560303, -2.190781354904175, -0.003579073818400502, 'Yes', '']]

npu_dict_aten = {'op_name': ['Aten__native_batch_norm_legit_functional.default_0_forward_input.0',
        'Aten__native_batch_norm_legit_functional.default_0_forward_input.1',
        'Aten__native_batch_norm_legit_functional.default_0_forward_input.2',
        'Aten__native_batch_norm_legit_functional.default_0_forward_input.3',
        'Aten__native_batch_norm_legit_functional.default_0_forward_input.4',
        'Aten__native_batch_norm_legit_functional.default_0_forward_output.0',
        'Aten__native_batch_norm_legit_functional.default_0_forward_output.1',
        'Aten__native_batch_norm_legit_functional.default_0_forward_output.2',
        'Aten__native_batch_norm_legit_functional.default_0_forward_output.3',
        'Aten__native_batch_norm_legit_functional.default_0_forward_output.4'],
        'input_struct': [('torch.float16', [256, 256, 14, 14]), ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256])],
        'output_struct': [('torch.float16', [256, 256, 14, 14]), ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256])],
        'summery': [[139.625, -127.5625, -0.0103607177734375],
        [2.5276029109954834, -2.1788690090179443, -0.0008259844034910202],
        [2.472219944000244, -2.845968723297119, -0.008756577968597412],
        [2.763145923614502, -3.398397922515869, -0.052132632583379745],
        [2.673110008239746, -3.149275064468384, 0.01613386906683445],
        [13.5546875, -10.640625, -0.008758544921875],
        [0.30550330877304077, -0.24485322833061218, -0.010361209511756897],
        [623.9192504882812, 432.96826171875, 520.2276611328125],
        [2.4797861576080322, -3.055997371673584, -0.04795549064874649],
        [61.7945556640625, 42.59713363647461, 52.03831481933594]]}

bench_dict_functional = {'op_name': ['Functional_batch_norm_0_forward_input.0', 'Functional_batch_norm_0_forward_input.1',
 'Functional_batch_norm_0_forward_input.2', 'Functional_batch_norm_0_forward_input.3',
  'Functional_batch_norm_0_forward_input.4', 'Functional_batch_norm_0_forward_output'],
   'input_struct': [('torch.float32', [256, 256, 14, 14]), ('torch.float32', [256]), ('torch.float32', [256]),
    ('torch.float32', [256]), ('torch.float32', [256])],
    'output_struct': [('torch.float32', [256, 256, 14, 14])],
    'summery': [[3.061628818511963, -3.22507381439209, 3.634914173744619e-05],
     [0.0005779837374575436, -0.0006301702815108001, 3.634906533989124e-06],
      [0.9338104128837585, 0.9277191162109375, 0.930335283279419],
       [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
        [5.397906303405762, -5.796811580657959, 2.5283952709287405e-10]]
}

aten_result = [['Aten__native_batch_norm_legit_functional.default_0_forward_input.0', 'Functional_batch_norm_0_forward_input.0', 'torch.float16', 'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], ' ', ' ', ' ', ' ', ' ', 139.625, -127.5625, -0.0103607177734375, 3.061628818511963, -3.22507381439209, 3.634914173744619e-05, 'Yes', ''],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_input.1', 'Functional_batch_norm_0_forward_input.1', 'torch.float32', 'torch.float32', [256], [256], ' ', ' ', ' ', ' ', ' ', 2.5276029109954834, -2.1788690090179443, -0.0008259844034910202, 0.0005779837374575436, -0.0006301702815108001, 3.634906533989124e-06, 'Yes', ''],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_input.2', 'Functional_batch_norm_0_forward_input.2', 'torch.float32', 'torch.float32', [256], [256], ' ', ' ', ' ', ' ', ' ', 2.472219944000244, -2.845968723297119, -0.008756577968597412, 0.9338104128837585, 0.9277191162109375, 0.930335283279419, 'Yes', ''],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_input.3', 'Functional_batch_norm_0_forward_input.3', 'torch.float32', 'torch.float32', [256], [256], ' ', ' ', ' ', ' ', ' ', 2.763145923614502, -3.398397922515869, -0.052132632583379745, 1.0, 1.0, 1.0, 'Yes', ''],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_input.4', 'Functional_batch_norm_0_forward_input.4', 'torch.float32', 'torch.float32', [256], [256], ' ', ' ', ' ', ' ', ' ', 2.673110008239746, -3.149275064468384, 0.01613386906683445, 0.0, 0.0, 0.0, 'Yes', ''],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_output.0', 'Functional_batch_norm_0_forward_output', 'torch.float16', 'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], ' ', ' ', ' ', ' ', ' ', 13.5546875, -10.640625, -0.008758544921875, 5.397906303405762, -5.796811580657959, 2.5283952709287405e-10, 'Yes', ''],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_output.1', 'Nan', 'torch.float32', 'Nan', [256], 'Nan', ' ', ' ', ' ', ' ', ' ', 0.30550330877304077, -0.24485322833061218, -0.010361209511756897, 'Nan', 'Nan', 'Nan', 'Yes', ''],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_output.2', 'Nan', 'torch.float32', 'Nan', [256], 'Nan', ' ', ' ', ' ', ' ', ' ', 623.9192504882812, 432.96826171875, 520.2276611328125, 'Nan', 'Nan', 'Nan', 'Yes', ''],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_output.3', 'Nan', 'torch.float32', 'Nan', [256], 'Nan', ' ', ' ', ' ', ' ', ' ', 2.4797861576080322, -3.055997371673584, -0.04795549064874649, 'Nan', 'Nan', 'Nan', 'Yes', ''],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_output.4', 'Nan', 'torch.float32', 'Nan', [256], 'Nan', ' ', ' ', ' ', ' ', ' ', 61.7945556640625, 42.59713363647461, 52.03831481933594, 'Nan', 'Nan', 'Nan', 'Yes', '']
    ]


class TestUtilsMethods(unittest.TestCase):
    def test_correct_data(self):
        input_1 = 'NAN'
        result_1 = compare.correct_data(input_1)
        self.assertEqual(result_1, 'NAN')
        input_2 = '0.99999'
        result_2 = compare.correct_data(input_2)
        self.assertEqual(result_2, '0.99999')
        input_3 = '0.999991'
        result_3 = compare.correct_data(input_3)
        self.assertEqual(result_3, '1.0')

    def test_cosine_similarity_when_all_result_less_than_epsilon(self):
        n_value = np.array([0, 0, 0])
        b_value = np.array([0, 0, 0])
        result, message = compare.cosine_similarity(n_value, b_value)
        self.assertEqual(result, '1.0')
        self.assertEqual(message, '')

    def test_cosine_similarity_when_only_npu_result_less_than_epsilon(self):
        n_value = np.array([0, 0, 0])
        b_value = np.array([1, 2, 3])
        result, message = compare.cosine_similarity(n_value, b_value)
        self.assertEqual(result, CompareConst.NAN)
        self.assertEqual(message, 'Cannot compare by Cosine Similarity, All the data is Zero in npu dump data.')

    def test_cosine_similarity_when_only_bench_result_less_than_epsilon(self):
        n_value = np.array([1, 2, 3])
        b_value = np.array([0, 0, 0])
        result, message = compare.cosine_similarity(n_value, b_value)
        self.assertEqual(result, CompareConst.NAN)
        self.assertEqual(message, 'Cannot compare by Cosine Similarity, All the data is Zero in Bench dump data.')

    def test_cosine_similarity_when_all_result_greater_than_epsilon_with_no_nan(self):
        n_value = np.array([1, 2, 3])
        b_value = np.array([1, 2, 3])
        result, message = compare.cosine_similarity(n_value, b_value)
        
        self.assertEqual(result, '1.0')
        self.assertEqual(message, '')

    def test_cosine_similarity_when_all_result_greater_than_epsilon_with_nan(self):
        n_value = np.array([1, 2, np.nan])
        b_value = np.array([1, 2, 3])
        result, message = compare.cosine_similarity(n_value, b_value)
        self.assertEqual(result, CompareConst.NAN)
        self.assertEqual(message, 'Cannot compare by Cosine Similarity, the dump data has NaN.')

    def test_get_rmse_when_rmse_is_nan(self):
        n_value = np.array([1, 2, np.nan])
        b_value = np.array([1, 2, 3])
        rmse, message = compare.get_rmse(n_value, b_value)
        self.assertEqual(rmse, CompareConst.NAN)
        self.assertEqual(message, "")

    def test_get_mape_when_mape_is_nan(self):
        n_value = np.array([1, 2, np.nan])
        b_value = np.array([1, 2, 3])
        mape, message = compare.get_mape(n_value, b_value)
        self.assertEqual(mape, CompareConst.NAN)
        self.assertEqual(message, "")

    def test_get_max_relative_err_when_max_relative_is_nan(self):
        n_value = np.array([1, 2, np.nan])
        b_value = np.array([1, 2, 3])
        max_relative_err, message = compare.get_max_relative_err(n_value, b_value)
        self.assertEqual(max_relative_err, CompareConst.NAN)
        self.assertEqual(message, 'Cannot compare by MaxRelativeError, the data contains nan in dump data.')

    def test_get_max_relative_err_when_max_relative_is_not_nan(self):
        n_value = np.array([1, 2, 3])
        b_value = np.array([1, 2, 3])
        max_relative_err, message = compare.get_max_relative_err(n_value, b_value)
        self.assertEqual(max_relative_err, "0.000000000000")
        self.assertEqual(message, "")

    def test_check_graph_mode(self):
        op1 = "Aten"
        op2 = "torch"
        self.assertTrue(compare.check_graph_mode(op1, op2))
        self.assertTrue(compare.check_graph_mode(op2, op1))
        self.assertFalse(compare.check_graph_mode(op1, op1))
        self.assertFalse(compare.check_graph_mode(op2, op2))

    def test_check_op(self):
        fuzzy_match = False
        result = compare.check_op(npu_dict, bench_dict, fuzzy_match)
        self.assertEqual(result, True)

    def test_merge_tensor(self):
        op_dict = compare.merge_tensor(tensor_list)
        self.assertEqual(op_dict, result_op_dict)

    def test_read_op(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        pkl_dir = os.path.join(base_dir, "resources/compare/npu_test.pkl")
 
        npu_ops_queue = []
        npu_pkl_handle = open(pkl_dir, "r")
        stack_mode = False
        result = compare.read_op(npu_ops_queue, npu_pkl_handle, stack_mode)
        self.assertEqual(result, True)


    def test_match_op(self):
        fuzzy_match = False
        a, b = compare.match_op([npu_dict], [bench_dict], fuzzy_match)
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_get_accuracy(self):
        result = []
        compare.get_accuracy(result, npu_dict, bench_dict)
        
        self.assertEqual(result, o_result)

    def test_get_accuracy_graph_mode(self):
        result = []
        compare.get_accuracy(result, npu_dict_aten, bench_dict_functional)
        self.assertEqual(result, aten_result)
