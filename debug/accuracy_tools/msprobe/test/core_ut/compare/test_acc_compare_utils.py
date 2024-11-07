# coding=utf-8
import os
import json
import shutil
import unittest
import argparse
from msprobe.core.compare.utils import extract_json, rename_api, read_op, op_item_parse, \
    check_and_return_dir_contents, resolve_api_special_parameters, get_rela_diff_summary_mode, \
    get_accuracy, get_un_match_accuracy, merge_tensor, _compare_parser
from msprobe.core.common.utils import CompareException
from msprobe.core.common.const import Const


# test_read_op_1
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
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'md5':'00000000',
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063, 'data_name': '-1',
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_0.0.forward.input.0'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'md5':'00000000',
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481, 'data_name': '-1',
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_0.0.forward.input.1'},
    {'full_op_name': 'Tensor.add_0.0.forward.input.alpha', 'dtype': "<class 'float'>", 'shape': '[]', 'md5': '0dae4479',
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1', 'type': 'float', 'value': -0.1},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'md5':'00000000',
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063, 'data_name': '-1',
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_0.0.forward.output.0'}]

# test_read_op_1
op_data_b = {
    'input': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
               'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
               'Norm': 2.2533628940582275, 'requires_grad': True},
              {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
               'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
               'Norm': 0.02844562754034996, 'requires_grad': False}],
    'output': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
                'Norm': 2.2533628940582275, 'requires_grad': True}]}
op_name_b = "Tensor.add_0.0.backward"
op_result_b = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'data_name': '-1', 'md5': '00000000',
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_0.0.backward.input.0'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'data_name': '-1', 'md5': '00000000',
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_0.0.backward.input.1'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3], 'data_name': '-1', 'md5': '00000000',
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_0.0.backward.output.0'}]


# test_op_item_parse
parse_item = [
    {'Max': 4097.0, 'Mean': 820.2, 'Min': 0.0, 'Norm': 4097.0, 'dtype': 'torch.int64', 'requires_grad': False, 'shape': [5], 'type': 'torch.Tensor'},
    {'type': 'int', 'value': 0},
    {'type': 'slice', 'value': [None, None, None]}
]
parse_op_name = 'Distributed.broadcast.0.forward.input'
parse_index = None
parse_item_list = None
parse_top_bool = True
o_result_parse = [
    {'Max': 4097.0, 'Mean': 820.2, 'Min': 0.0, 'Norm': 4097.0, 'dtype': 'torch.int64', 'requires_grad': False,
     'shape': [5], 'type': 'torch.Tensor', 'full_op_name': 'Distributed.broadcast.0.forward.input.0',
     'data_name': '-1', 'md5': '00000000'},
    {'full_op_name': 'Distributed.broadcast.0.forward.input.1', 'dtype': "<class 'int'>", 'shape': '[]',
     'md5': 'f4dbdf21', 'Max': 0, 'Min': 0, 'Mean': 0, 'Norm': 0, 'data_name': '-1', 'type': 'int', 'value': 0},
    {'Max': None, 'Mean': None, 'Min': None, 'Norm': None, 'data_name': '-1', 'dtype': 'slice', 'type': 'slice',
     'full_op_name': 'Distributed.broadcast.0.forward.input.2', 'md5': '5fbbe87f', 'shape': '(3,)',
     'value': [None, None, None]}
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
     'full_op_name': 'Tensor.add_.0.forward.input.0'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_.0.forward.input.1'},
    {'full_op_name': 'Tensor.add_.0.forward.input.alpha.0', 'dtype': "<class 'float'>", "shape": '[]', 'md5': None,
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_.0.forward.output.0'}
]
result_op_dict = {'op_name': ['Tensor.add_.0.forward.input.0', 'Tensor.add_.0.forward.input.1',
                              'Tensor.add_.0.forward.input.alpha.0', 'Tensor.add_.0.forward.output.0'],
                  'input_struct': [('torch.float32', [16, 1, 3, 3]), ('torch.float32', [16, 1, 3, 3]),
                                   ("<class 'float'>", '[]')],
                  'output_struct': [('torch.float32', [16, 1, 3, 3])],
                  'summary': [[0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275],
                              [0.003992878366261721, -0.008102823048830032, -0.0002002553956117481, 0.02844562754034996],
                              [-0.1, -0.1, -0.1, -0.1],
                              [0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275]],
                  'stack_info': []}

tensor_list_md5 = [
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
     'Norm': 0.02844562754034996, 'requires_grad': False, 'full_op_name': 'Tensor.add_.0.forward.input.0', 'md5': 1},
    {'full_op_name': 'Tensor.add_.0.forward.kwargs.alpha.0', 'dtype': "<class 'float'>", "shape": '[]', 'md5': None,
     'Max': -0.1, 'Min': -0.1, 'Mean': -0.1, 'Norm': -0.1, 'data_name': '-1'},
    {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
     'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
     'Norm': 2.2533628940582275, 'requires_grad': True, 'full_op_name': 'Tensor.add_.0.forward.output.0', 'md5': 2}
]
result_op_dict_md5 = {'op_name': ['Tensor.add_.0.forward.input.0', 'Tensor.add_.0.forward.kwargs.alpha.0',
                                  'Tensor.add_.0.forward.output.0'],
                      'input_struct': [('torch.float32', [16, 1, 3, 3], 1)],
                      'kwargs_struct': [("<class 'float'>", '[]', None)],
                      'output_struct': [('torch.float32', [16, 1, 3, 3], 2)],
                      'summary': [[0.003992878366261721, -0.008102823048830032, -0.0002002553956117481, 0.02844562754034996],
                                  [-0.1, -0.1, -0.1, -0.1],
                                  [0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275]],
                      'stack_info': []}


base_dir1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_utils1')
base_dir2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_utils2')


def create_json_files(base_dir):
    file_names = ['dump.json', 'stack.json', 'construct.json']

    for file_name in file_names:
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w') as f:
            json.dump({}, f)


def create_rank_dirs(base_dir):
    folder_names = ['rank0', 'rank1']

    for folder_name in folder_names:
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)


class TestUtilsMethods(unittest.TestCase):

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        _compare_parser(self.parser)

        os.makedirs(base_dir1, mode=0o750, exist_ok=True)
        os.makedirs(base_dir2, mode=0o750, exist_ok=True)

    def tearDown(self):
        if os.path.exists(base_dir1):
            shutil.rmtree(base_dir1)
        if os.path.exists(base_dir2):
            shutil.rmtree(base_dir2)

    def test_extract_json_1(self):
        create_json_files(base_dir1)
        result = extract_json(base_dir1, stack_json=False)
        self.assertEqual(result, os.path.join(base_dir1, 'dump.json'))

        result = extract_json(base_dir1, stack_json=True)
        self.assertEqual(result, os.path.join(base_dir1, 'stack.json'))

    def test_check_and_return_dir_contents(self):
        create_rank_dirs(base_dir2)
        result = check_and_return_dir_contents(base_dir2, 'rank')
        self.assertEqual(set(result), set(['rank0', 'rank1']))

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

    def test_read_op_back(self):
        result = read_op(op_data_b, op_name_b)
        self.assertEqual(result, op_result_b)

    def test_op_item_parse(self):
        result = op_item_parse(parse_item, parse_op_name)
        self.assertEqual(result, o_result_parse)
    
    def test_op_item_parse_max_depth(self):
        with self.assertRaises(CompareException) as context:
            op_item_parse(parse_item, parse_op_name, depth=11)
        self.assertEqual(context.exception.code, CompareException.RECURSION_LIMIT_ERROR)

    def test_resolve_api_special_parameters(self):
        item_list = []
        resolve_api_special_parameters(data_dict, full_op_name, item_list)
        self.assertEqual(item_list, o_result_api_special)

    def test_get_rela_diff_summary_mode_float_or_int(self):
        result_item = [0] * 14
        err_msg = ''
        npu_summary_data = [1, 1, 1, 1]
        bench_summary_data = [1, 1, 1, 1]
        result_item, accuracy_check, err_msg = get_rela_diff_summary_mode(result_item, npu_summary_data,
                                                                          bench_summary_data, err_msg)
        self.assertEqual(result_item, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '0.0%', '0.0%', '0.0%', '0.0%'])
        self.assertEqual(accuracy_check, '')
        self.assertEqual(err_msg, '')

    def test_get_rela_diff_summary_mode_bool(self):
        result_item = [0] * 14
        err_msg = ''
        npu_summary_data = [True, True, True, True]
        bench_summary_data = [True, True, True, True]
        result_item, accuracy_check, err_msg = get_rela_diff_summary_mode(result_item, npu_summary_data,
                                                                          bench_summary_data, err_msg)
        self.assertEqual(result_item, [0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
        self.assertEqual(accuracy_check, '')
        self.assertEqual(err_msg, '')

    def test_get_rela_diff_summary_mode_nan(self):
        result_item = [0] * 14
        err_msg = ''
        npu_summary_data = [float('nan')]
        bench_summary_data = [float('nan')]
        result_item, accuracy_check, err_msg = get_rela_diff_summary_mode(result_item, npu_summary_data,
                                                                          bench_summary_data, err_msg)
        self.assertEqual(result_item, [0, 0, 0, 0, 0, 0, 'Nan', 0, 0, 0, 'Nan', 0, 0, 0])
        self.assertEqual(accuracy_check, '')
        self.assertEqual(err_msg, '')


    def test_get_accuracy(self):
        result = []
        get_accuracy(result, npu_dict, bench_dict, dump_mode=Const.SUMMARY)
        self.assertEqual(result, o_result)

    def test_get_un_match_accuracy_md5(self):
        result = []
        get_un_match_accuracy(result, npu_dict, dump_mode=Const.MD5)
        self.assertEqual(result, o_result_unmatch_1)

    def test_get_un_match_accuracy_summary(self):
        result = []
        get_un_match_accuracy(result, npu_dict, dump_mode=Const.SUMMARY)
        self.assertEqual(result, o_result_unmatch_2)

    def test_get_un_match_accuracy_all(self):
        result = []
        get_un_match_accuracy(result, npu_dict, dump_mode=Const.ALL)
        self.assertEqual(result, o_result_unmatch_3)

    def test_merge_tensor_summary(self):
        op_dict = merge_tensor(tensor_list, dump_mode=Const.SUMMARY)
        self.assertEqual(op_dict, result_op_dict)

    def test_merge_tensor_md5(self):
        op_dict = merge_tensor(tensor_list_md5, dump_mode=Const.MD5)
        self.assertEqual(op_dict, result_op_dict_md5)

    def test_compare_parser_1(self):
        test_args = ["-i", "input.json", "-o", "output.json", "-s", "-c", "-f"]
        args = self.parser.parse_args(test_args)

        self.assertEqual(args.input_path, "input.json")
        self.assertEqual(args.output_path, "output.json")
        self.assertTrue(args.stack_mode)
        self.assertTrue(args.compare_only)
        self.assertTrue(args.fuzzy_match)

    def test_compare_parser_2(self):
        test_args = ["-i", "input.json"]

        with self.assertRaises(SystemExit):  # argparse 会抛出 SystemExit
            self.parser.parse_args(test_args)

    def test_compare_parser_3(self):
        test_args = ["-i", "input.json", "-o", "output.json", "-cm", "cell_mapping.txt", "-dm",
                     "data_mapping.txt", "-lm", "layer_mapping.txt"]
        args = self.parser.parse_args(test_args)

        self.assertEqual(args.cell_mapping, "cell_mapping.txt")
        self.assertIsNone(args.api_mapping)  # 默认值应为 None
        self.assertEqual(args.data_mapping, "data_mapping.txt")
        self.assertEqual(args.layer_mapping, "layer_mapping.txt")
