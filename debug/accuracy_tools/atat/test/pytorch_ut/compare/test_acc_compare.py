# coding=utf-8
import unittest
from msprobe.pytorch.compare import acc_compare as compare
import pandas as pd

npu_dict = {'op_name': ['Functional_conv2d_0_forward_input.0', 'Functional_conv2d_0_forward_input.1',
                        'Functional_conv2d_0_forward_input.2', 'Functional_conv2d_0_forward_output'],
            'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                             ('torch.float32', [16])],
            'output_struct': [('torch.float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

bench_dict = {'op_name': ['Functional_conv2d_0_forward_input.0', 'Functional_conv2d_0_forward_input.1',
                          'Functional_conv2d_0_forward_input.2', 'Functional_conv2d_0_forward_output'],
              'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                               ('torch.float32', [16])],
              'output_struct': [('torch.float32', [1, 16, 28, 28])],
              'summery': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                          [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                          [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                          [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

tensor_list = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'Max': 0.33033010363578796,
     'Min': -0.331031858921051,'Mean': -0.030964046716690063, 'Norm': 2.2533628940582275, 'requires_grad': True,
     'full_op_name': 'Tensor.add_.0.forward_input.0'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_.0.forward_input.1'},
    {'full_op_name': 'Tensor.add_.0.forward_input.alpha.0', 'dtype': "<class 'float'>", "shape": '[]', 'md5': None,
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_.0.forward_output.0'}
]

result_op_dict = {'op_name': ['Tensor.add_.0.forward_input.0', 'Tensor.add_.0.forward_input.1',
                              'Tensor.add_.0.forward_input.alpha.0', 'Tensor.add_.0.forward_output.0'],
                  'input_struct': [('torch.float32', [16, 1, 3, 3]), ('torch.float32', [16, 1, 3, 3]),
                                   ("<class 'float'>", '[]')],
                  'output_struct': [('torch.float32', [16, 1, 3, 3])],
                  'summary': [[0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275],
                              [0.003992878366261721, -0.008102823048830032, -0.0002002553956117481, 0.02844562754034996],
                              [-0.1, -0.1, -0.1, -0.1],
                              [0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275]],
                  'stack_info': []}

o_result = [
    ['Functional_conv2d_0_forward_input.0', 'Functional_conv2d_0_forward_input.0', 'torch.float32', 'torch.float32',
     [1, 1, 28, 28], [1, 1, 28, 28], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 3.029174327850342, -2.926689624786377,
     -0.06619918346405029, 3.029174327850342, -2.926689624786377, -0.06619918346405029, '', '', 'None'],
    ['Functional_conv2d_0_forward_input.1', 'Functional_conv2d_0_forward_input.1', 'torch.float32', 'torch.float32',
     [16, 1, 5, 5], [16, 1, 5, 5], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 0.19919930398464203, -0.19974489510059357,
     0.006269412115216255, 0.19919930398464203, -0.19974489510059357, 0.006269412115216255, '', '', 'None'],
    ['Functional_conv2d_0_forward_input.2', 'Functional_conv2d_0_forward_input.2', 'torch.float32', 'torch.float32',
     [16], [16], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 0.19734230637550354, -0.18177609145641327, 0.007903944700956345,
     0.19734230637550354, -0.18177609145641327, 0.007903944700956345, '', '', 'None'],
    ['Functional_conv2d_0_forward_output', 'Functional_conv2d_0_forward_output', 'torch.float32', 'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 2.1166646480560303, -2.190781354904175,
     -0.003579073818400502, 2.1166646480560303, -2.190781354904175, -0.003579073818400502, '', '', 'None']]

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
                 'input_struct': [('torch.float16', [256, 256, 14, 14]), ('torch.float32', [256]),
                                  ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256])],
                 'output_struct': [('torch.float16', [256, 256, 14, 14]), ('torch.float32', [256]),
                                   ('torch.float32', [256]), ('torch.float32', [256]), ('torch.float32', [256])],
                 'summary': [[139.625, -127.5625, -0.0103607177734375],
                             [2.5276029109954834, -2.1788690090179443, -0.0008259844034910202],
                             [2.472219944000244, -2.845968723297119, -0.008756577968597412],
                             [2.763145923614502, -3.398397922515869, -0.052132632583379745],
                             [2.673110008239746, -3.149275064468384, 0.01613386906683445],
                             [13.5546875, -10.640625, -0.008758544921875],
                             [0.30550330877304077, -0.24485322833061218, -0.010361209511756897],
                             [623.9192504882812, 432.96826171875, 520.2276611328125],
                             [2.4797861576080322, -3.055997371673584, -0.04795549064874649],
                             [61.7945556640625, 42.59713363647461, 52.03831481933594]]}

bench_dict_functional = {
    'op_name': ['Functional_batch_norm_0_forward_input.0', 'Functional_batch_norm_0_forward_input.1',
                'Functional_batch_norm_0_forward_input.2', 'Functional_batch_norm_0_forward_input.3',
                'Functional_batch_norm_0_forward_input.4', 'Functional_batch_norm_0_forward_output'],
    'input_struct': [('torch.float32', [256, 256, 14, 14]), ('torch.float32', [256]), ('torch.float32', [256]),
                     ('torch.float32', [256]), ('torch.float32', [256])],
    'output_struct': [('torch.float32', [256, 256, 14, 14])],
    'summary': [[3.061628818511963, -3.22507381439209, 3.634914173744619e-05],
                [0.0005779837374575436, -0.0006301702815108001, 3.634906533989124e-06],
                [0.9338104128837585, 0.9277191162109375, 0.930335283279419],
                [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
                [5.397906303405762, -5.796811580657959, 2.5283952709287405e-10]]
}

aten_result = [
    ['Aten__native_batch_norm_legit_functional.default_0_forward_input.0', 'Functional_batch_norm_0_forward_input.0',
     'torch.float16', 'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 136.56337118148804, -124.33742618560791,
     -0.010397066915174946, ' ', '4460.480981749501%', '3855.335826136584%', '28603.33536971545%', ' ', 139.625,
     -127.5625, -0.0103607177734375, 3.061628818511963, -3.22507381439209, 3.634914173744619e-05, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_input.1', 'Functional_batch_norm_0_forward_input.1',
     'torch.float32', 'torch.float32', [256], [256], 2.527024927258026, -2.1782388387364335, -0.0008296193100250093,
     ' ', '437213.84590749856%', '345658.76916858414%', '22823.676544842117%', ' ', 2.5276029109954834,
     -2.1788690090179443, -0.0008259844034910202, 0.0005779837374575436, -0.0006301702815108001, 3.634906533989124e-06,
     'Warning', 'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_input.2', 'Functional_batch_norm_0_forward_input.2',
     'torch.float32', 'torch.float32', [256], [256], 1.5384095311164856, -3.7736878395080566, -0.9390918612480164, ' ',
     '164.74538192025793%', '406.7705163736246%', '100.94122819224167%', ' ', 2.472219944000244, -2.845968723297119,
     -0.008756577968597412, 0.9338104128837585, 0.9277191162109375, 0.930335283279419, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_input.3', 'Functional_batch_norm_0_forward_input.3',
     'torch.float32', 'torch.float32', [256], [256], 1.763145923614502, -4.398397922515869, -1.0521326325833797, ' ',
     '176.3145923614502%', '439.8297922515869%', '105.21326325933797%', ' ', 2.763145923614502, -3.398397922515869,
     -0.052132632583379745, 1.0, 1.0, 1.0, 'Warning', 'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_input.4', 'Functional_batch_norm_0_forward_input.4',
     'torch.float32', 'torch.float32', [256], [256], 2.673110008239746, -3.149275064468384, 0.01613386906693445, ' ',
     'N/A', 'N/A', 'N/A', ' ', 2.673110008239746, -3.149275064468384, 0.01613386906683445, 0.0, 0.0, 0.0, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_output.0', 'Functional_batch_norm_0_forward_output',
     'torch.float16', 'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 8.156781196594238, -4.843813419342041,
     -0.008758545174714527, ' ', '151.11009228611078%', '83.55995967687207%', '3464072756.115108%', ' ', 13.5546875,
     -10.640625, -0.008758544921875, 5.397906303405762, -5.796811580657959, 2.5283952709287405e-10, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_output.1', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', 0.30550330877304077, -0.24485322833061218, -0.010361209511756897, 'Nan', 'Nan', 'Nan',
     'Yes', '', None],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_output.2', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', 623.9192504882812, 432.96826171875, 520.2276611328125, 'Nan', 'Nan', 'Nan',
     'Yes', '', None],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_output.3', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', 2.4797861576080322, -3.055997371673584, -0.04795549064874649, 'Nan', 'Nan', 'Nan',
     'Yes', '', None],
    ['Aten__native_batch_norm_legit_functional.default_0_forward_output.4', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', 61.7945556640625, 42.59713363647461, 52.03831481933594, 'Nan', 'Nan', 'Nan',
     'Yes', '', None]]

highlight_dict = {'red_rows': [], 'yellow_rows': []}

num_0, num_1, num_2, num_3 = 0, 1, 2, 3
summary_line_input = ['Functional_batch_norm_0_forward_input.0', 'Functional_batch_norm_0_forward_input.0',
                      'torch.float16',
                      'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0.01, 0, 0, 0, 1, 1, 1, 1, 1.01, 1, 1, 1,
                      'Yes', '']
summary_line_1 = ['Functional_batch_norm_0_forward_output.0', 'Functional_batch_norm_0_forward_output.0',
                  'torch.float16',
                  'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 10, 0, 0, 0, 2, 0, 1, 1, 1, 1, 1, 1,
                  'Warning', '']
summary_line_2 = ['Functional_batch_norm_0_forward_output.1', 'Functional_batch_norm_0_forward_output.1',
                  'torch.float16',
                  'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0.02, 0, 0, 0, 0.12, 0, 1, 1, 0.1, 1, 1, 1,
                  'Warning', '']
summary_line_3 = ['Functional_batch_norm_0_forward_output.2', 'Functional_batch_norm_0_forward_output.2',
                  'torch.float16',
                  'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0, 0, 0, 0, 2, 0, 1, 1, 1, 1, 1, 1,
                  'Warning', '']
line_input = ['Functional_batch_norm_0_forward_input.0', 'Functional_batch_norm_0_forward_input.0', 'torch.float16',
              'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 1, 1, 1, 0.95, 1, 1, 1, 1, 1, 1.01, 1, 1, 1,
              'Yes', '']
line_1 = ['Functional_batch_norm_0_forward_output.0', 'Functional_batch_norm_0_forward_output.0', 'torch.float16',
          'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0.8, 1, 1, 0.59, 1, 'nan', 0, 1, 1, 19, 1, 1, 1,
          'Warning', '']
line_2 = ['Functional_batch_norm_0_forward_output.1', 'Functional_batch_norm_0_forward_output.1', 'torch.float16',
          'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0.9, 1, 1, 0.8, 1, 0, 0.12, 0, 1, 1, 0.1, 1, 1, 1,
          'Warning', '']
line_3 = ['Functional_batch_norm_0_forward_output.2', 'Functional_batch_norm_0_forward_output.2', 'torch.float16',
          'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0.8, 1.1e+10, 1, 0.85, 1, 9, 0.12, 0, 1, 1, 0.1, 1,
          1, 1, 'Warning', '']

op_data = {
    'input_args': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                    'Max': 0.33033010363578796, 'Min': -0.331031858921051,'Mean': -0.030964046716690063,
                    'Norm': 2.2533628940582275, 'requires_grad': True},
                   {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                    'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
                    'Norm': 0.02844562754034996, 'requires_grad': False}],
    'input_kwargs': {'alpha': {'type': 'float', 'value': -0.1}},
    'output': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                'Max': 0.33033010363578796, 'Min': -0.331031858921051,'Mean': -0.030964046716690063,
                'Norm': 2.2533628940582275, 'requires_grad': True}]}

op_name = "Tensor.add_0.0.forward"

op_result = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051,'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_.0.forward_input.0'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032,'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_.0.forward_input.1'},
    {'full_op_name': 'Tensor.add_.0.forward_input.alpha.0', 'dtype': "<class 'float'>", "shape": '[]', 'md5': None,
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051,'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_.0.forward_output.0'}]


class TestUtilsMethods(unittest.TestCase):

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
        op_dict = compare.merge_tensor(tensor_list, True, False)
        self.assertEqual(op_dict, result_op_dict)

    def test_read_op(self):
        result = compare.read_op(op_data, op_name)
        self.assertEqual(result, op_result)

    def test_match_op(self):
        fuzzy_match = False
        a, b = compare.match_op([npu_dict], [bench_dict], fuzzy_match)
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_get_accuracy(self):
        result = []
        compare.get_accuracy(result, npu_dict, bench_dict, highlight_dict)
        self.assertEqual(result, o_result)

    def test_get_accuracy_graph_mode(self):
        result = []
        compare.get_accuracy(result, npu_dict_aten, bench_dict_functional, highlight_dict)
        self.assertEqual(result, aten_result)

    def test_find_error_rows(self):
        summary_result = [summary_line_input, summary_line_1, summary_line_2, summary_line_3]
        highlight_dict = {'red_rows': [], 'yellow_rows': []}
        compare.find_error_rows(summary_result, 0, 1, highlight_dict, summary_compare=True)
        self.assertEqual(highlight_dict, {'red_rows': [], 'yellow_rows': []})

    def test_find_compare_result_error_rows(self):
        result = [line_input, line_1, line_2, line_3]
        result_df = pd.DataFrame(result)
        highlight_dict = {'red_rows': [], 'yellow_rows': []}
        compare.find_compare_result_error_rows(result_df, highlight_dict, False)
        self.assertEqual(highlight_dict, {'red_rows': [num_1, num_3], 'yellow_rows': [num_2]})

    def test_rename_api(self):
        test_name_1 = "Distributed.broadcast.0.forward.input.0"
        expect_name_1 = "Distributed.broadcast.input.0"
        actual_name_1 = compare.rename_api(test_name_1, "forward")
        self.assertEqual(actual_name_1, expect_name_1)

        test_name_2 = "Torch.sum.0.backward.output.0"
        expect_name_2 = "Torch.sum.output.0"
        actual_name_2 = compare.rename_api(test_name_2, "backward")
        self.assertEqual(actual_name_2, expect_name_2)
