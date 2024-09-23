# coding=utf-8
import unittest
from msprobe.core.compare.utils import rename_api, read_op, op_item_parse, resolve_api_special_parameters, \
    get_accuracy, get_un_match_accuracy, merge_tensor


# test_read_op
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
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_0.0.forward.input.0'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_0.0.forward.input.1'},
    {'full_op_name': 'Tensor.add_0.0.forward.input.alpha.0', 'dtype': "<class 'float'>", 'shape': '[]', 'md5': None,
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_0.0.forward.output.0'}]


# test_op_item_parse
parse_item = [
    {'Max': 4097.0, 'Mean': 820.2, 'Min': 0.0, 'Norm': 4097.0, 'dtype': 'torch.int64', 'requires_grad': False, 'shape': [5], 'type': 'torch.Tensor'},
    {'type': 'int', 'value': 0}
]
parse_op_name = 'Distributed.broadcast.0.forward.input'
parse_index = None
parse_item_list = None
parse_top_bool = True
o_result_parse = [
    {'Max': 4097.0, 'Mean': 820.2, 'Min': 0.0, 'Norm': 4097.0, 'dtype': 'torch.int64', 'requires_grad': False, 'shape': [5], 'type': 'torch.Tensor', 'full_op_name': 'Distributed.broadcast.0.forward.input.0'},
    {'full_op_name': 'Distributed.broadcast.0.forward.input.1', 'dtype': "<class 'int'>", 'shape': '[]', 'md5': None, 'Max': 0, 'Min': 0, 'Mean': 0, 'Norm': 0, 'data_name': '-1'}
]


# test_resolve_api_special_parameters
data_dict = {
    "last_hidden_state":
        {"type": "torch.Tensor", "dtype": "torch.bfloat16"},
    "loss":
        {"type": "torch.Tensor", "dtype": "torch.float32"}
}
full_op_name = "Tensor.add_0.0.forward.input.0"
o_result_api_special = [
    {"type": "torch.Tensor", "dtype": "torch.bfloat16", "full_op_name": "Tensor.add_0.0.forward.input.last_hidden_state.0"},
    {"type": "torch.Tensor", "dtype": "torch.float32", "full_op_name": "Tensor.add_0.0.forward.input.loss.0"}
]


# test_get_accuracy
npu_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                        'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
           'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                             ('torch.float32', [16])],
            'output_struct': [('torch.float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}
bench_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
             'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                               ('torch.float32', [16])],
              'output_struct': [('torch.float32', [1, 16, 28, 28])],
              'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                          [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                          [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                          [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}
highlight_dict = {'red_rows': [], 'yellow_rows': []}
o_result = [
    ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.0', 'torch.float32', 'torch.float32',
     [1, 1, 28, 28], [1, 1, 28, 28], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 3.029174327850342, -2.926689624786377,
     -0.06619918346405029, 3.029174327850342, -2.926689624786377, -0.06619918346405029, '', '', 'None'],
    ['Functional.conv2d.0.forward.input.1', 'Functional.conv2d.0.forward.input.1', 'torch.float32', 'torch.float32',
     [16, 1, 5, 5], [16, 1, 5, 5], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 0.19919930398464203, -0.19974489510059357,
     0.006269412115216255, 0.19919930398464203, -0.19974489510059357, 0.006269412115216255, '', '', 'None'],
    ['Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.input.2', 'torch.float32', 'torch.float32',
     [16], [16], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 0.19734230637550354, -0.18177609145641327, 0.007903944700956345,
     0.19734230637550354, -0.18177609145641327, 0.007903944700956345, '', '', 'None'],
    ['Functional.conv2d.0.forward.output', 'Functional.conv2d.0.forward.output', 'torch.float32', 'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 2.1166646480560303, -2.190781354904175,
     -0.003579073818400502, 2.1166646480560303, -2.190781354904175, -0.003579073818400502, '', '', 'None']]


# test_get_un_match_accuracy
o_result_unmatch_1 = [
    ['Functional.conv2d.0.forward.input.0', 'N/A', 'torch.float32', 'N/A', [1, 1, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A', 'None'],
    ['Functional.conv2d.0.forward.input.1', 'N/A', 'torch.float32', 'N/A', [16, 1, 5, 5], 'N/A', 'N/A', 'N/A', 'N/A', 'None'],
    ['Functional.conv2d.0.forward.input.2', 'N/A', 'torch.float32', 'N/A', [16], 'N/A', 'N/A', 'N/A', 'N/A', 'None'],
    ['Functional.conv2d.0.forward.output', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A', 'None']
]
o_result_unmatch_2 = [
    ['Functional.conv2d.0.forward.input.0', 'N/A', 'torch.float32', 'N/A', [1, 1, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 3.029174327850342, -2.926689624786377, -0.06619918346405029, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.forward.input.1', 'N/A', 'torch.float32', 'N/A', [16, 1, 5, 5], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.forward.input.2', 'N/A', 'torch.float32', 'N/A', [16], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None'],
    ['Functional.conv2d.0.forward.output', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 2.1166646480560303, -2.190781354904175, -0.003579073818400502, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None']
]
o_result_unmatch_3 = [
    ['Functional.conv2d.0.forward.input.0', 'N/A', 'torch.float32', 'N/A', [1, 1, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 3.029174327850342, -2.926689624786377, -0.06619918346405029, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None', '-1'],
    ['Functional.conv2d.0.forward.input.1', 'N/A', 'torch.float32', 'N/A', [16, 1, 5, 5], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 0.19919930398464203, -0.19974489510059357, 0.006269412115216255, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None', '-1'],
    ['Functional.conv2d.0.forward.input.2', 'N/A', 'torch.float32', 'N/A', [16], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 0.19734230637550354, -0.18177609145641327, 0.007903944700956345, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None', '-1'],
    ['Functional.conv2d.0.forward.output', 'N/A', 'torch.float32', 'N/A', [1, 16, 28, 28], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 2.1166646480560303, -2.190781354904175, -0.003579073818400502, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'No bench data matched.', 'None', '-1']
]


# test_merge_tensor
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


class TestUtilsMethods(unittest.TestCase):

    def test_rename_api_1(self):
        test_name_1 = "Distributed.broadcast.0.forward.input.0"
        expect_name_1 = "Distributed.broadcast.input.0"
        actual_name_1 = rename_api(test_name_1, "forward")
        self.assertEqual(actual_name_1, expect_name_1)

    def test_rename_api_2(self):
        test_name_2 = "Torch.sum.0.backward.output.0"
        expect_name_2 = "Torch.sum.output.0"
        actual_name_2 = rename_api(test_name_2, "backward")
        self.assertEqual(actual_name_2, expect_name_2)

    def test_read_op(self):
        result = read_op(op_data, op_name)
        self.assertEqual(result, op_result)

    def test_op_item_parse(self):
        result = op_item_parse(parse_item, parse_op_name, parse_index, parse_item_list, parse_top_bool)
        self.assertEqual(result, o_result_parse)

    def test_resolve_api_special_parameters(self):
        item_list = []
        resolve_api_special_parameters(data_dict, full_op_name, item_list)
        self.assertEqual(item_list, o_result_api_special)

    def test_get_accuracy(self):
        result = []
        get_accuracy(result, npu_dict, bench_dict, highlight_dict)
        self.assertEqual(result, o_result)

    def test_get_un_match_accuracy_1(self):
        result = []
        get_un_match_accuracy(result, npu_dict, md5_compare=True, summary_compare=False)
        self.assertEqual(result, o_result_unmatch_1)

    def test_get_un_match_accuracy_2(self):
        result = []
        get_un_match_accuracy(result, npu_dict, md5_compare=False, summary_compare=True)
        self.assertEqual(result, o_result_unmatch_2)

    def test_get_un_match_accuracy_3(self):
        result = []
        get_un_match_accuracy(result, npu_dict, md5_compare=False, summary_compare=False)
        self.assertEqual(result, o_result_unmatch_3)

    def test_merge_tensor(self):
        op_dict = merge_tensor(tensor_list, True, False)
        self.assertEqual(op_dict, result_op_dict)
