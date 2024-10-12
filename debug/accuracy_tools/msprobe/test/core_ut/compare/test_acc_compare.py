# coding=utf-8
import unittest
import pandas as pd
import os
import shutil
import json
import torch
import threading
from msprobe.core.compare.utils import get_accuracy
from msprobe.core.compare.highlight import find_error_rows, find_compare_result_error_rows
from msprobe.core.compare.acc_compare import Comparator
from msprobe.core.common.const import CompareConst, Const
from msprobe.pytorch.compare.pt_compare import PTComparator


npu_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                        'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
           'input_struct': [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]),
                             ('torch.float32', [16])],
            'output_struct': [('torch.float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

npu_dict2 = {'op_name': ['Functional.conv2d.1.forward.input.0', 'Functional.conv2d.0.forward.input.1',
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

npu_dict_aten = {'op_name': ['Aten__native_batch_norm_legit_functional.default_0_forward.input.0',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.input.1',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.input.2',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.input.3',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.input.4',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.0',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.1',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.2',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.3',
                             'Aten__native_batch_norm_legit_functional.default_0_forward.output.4'],
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
    'op_name': ['Functional_batch_norm_0_forward.input.0', 'Functional_batch_norm_0_forward.input.1',
                'Functional_batch_norm_0_forward.input.2', 'Functional_batch_norm_0_forward.input.3',
                'Functional_batch_norm_0_forward.input.4', 'Functional_batch_norm_0_forward.output'],
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
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.0', 'Functional_batch_norm_0_forward.input.0',
     'torch.float16', 'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 136.56337118148804, -124.33742618560791,
     -0.010397066915174946, ' ', '4460.480981749501%', '3855.335826136584%', '28603.33536971545%', ' ', 139.625,
     -127.5625, -0.0103607177734375, 3.061628818511963, -3.22507381439209, 3.634914173744619e-05, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.1', 'Functional_batch_norm_0_forward.input.1',
     'torch.float32', 'torch.float32', [256], [256], 2.527024927258026, -2.1782388387364335, -0.0008296193100250093,
     ' ', '437213.84590749856%', '345658.76916858414%', '22823.676544842117%', ' ', 2.5276029109954834,
     -2.1788690090179443, -0.0008259844034910202, 0.0005779837374575436, -0.0006301702815108001, 3.634906533989124e-06,
     'Warning', 'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.2', 'Functional_batch_norm_0_forward.input.2',
     'torch.float32', 'torch.float32', [256], [256], 1.5384095311164856, -3.7736878395080566, -0.9390918612480164, ' ',
     '164.74538192025793%', '406.7705163736246%', '100.94122819224167%', ' ', 2.472219944000244, -2.845968723297119,
     -0.008756577968597412, 0.9338104128837585, 0.9277191162109375, 0.930335283279419, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.3', 'Functional_batch_norm_0_forward.input.3',
     'torch.float32', 'torch.float32', [256], [256], 1.763145923614502, -4.398397922515869, -1.0521326325833797, ' ',
     '176.3145923614502%', '439.8397922515869%', '105.21326325833797%', ' ', 2.763145923614502, -3.398397922515869,
     -0.052132632583379745, 1.0, 1.0, 1.0, 'Warning', 'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.input.4', 'Functional_batch_norm_0_forward.input.4',
     'torch.float32', 'torch.float32', [256], [256], 2.673110008239746, -3.149275064468384, 0.01613386906683445, ' ',
     'N/A', 'N/A', 'N/A', ' ', 2.673110008239746, -3.149275064468384, 0.01613386906683445, 0.0, 0.0, 0.0, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.0', 'Functional_batch_norm_0_forward.output',
     'torch.float16', 'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 8.156781196594238, -4.843813419342041,
     -0.008758545174714527, ' ', '151.11009228611078%', '83.55995967687207%', '3464072756.115108%', ' ', 13.5546875,
     -10.640625, -0.008758544921875, 5.397906303405762, -5.796811580657959, 2.5283952709287405e-10, 'Warning',
     'Need double check api accuracy.', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.1', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', 0.30550330877304077, -0.24485322833061218, -0.010361209511756897, 'Nan', 'Nan', 'Nan',
     'Yes', '', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.2', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', 623.9192504882812, 432.96826171875, 520.2276611328125, 'Nan', 'Nan', 'Nan',
     'Yes', '', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.3', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', 2.4797861576080322, -3.055997371673584, -0.04795549064874649, 'Nan', 'Nan', 'Nan',
     'Yes', '', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.4', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', 61.7945556640625, 42.59713363647461, 52.03831481933594, 'Nan', 'Nan', 'Nan',
     'Yes', '', 'None']]

highlight_dict = {'red_rows': [], 'yellow_rows': []}

num_0, num_1, num_2, num_3 = 0, 1, 2, 3
summary_line_input = ['Functional_batch_norm_0_forward.input.0', 'Functional_batch_norm_0_forward.input.0',
                      'torch.float16',
                      'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0.01, 0, 0, 0, 1, 1, 1, 1, 1.01, 1, 1, 1,
                      'Yes', '']
summary_line_1 = ['Functional_batch_norm_0_forward.output.0', 'Functional_batch_norm_0_forward.output.0',
                  'torch.float16',
                  'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 10, 0, 0, 0, 2, 0, 1, 1, 1, 1, 1, 1,
                  'Warning', '']
summary_line_2 = ['Functional_batch_norm_0_forward.output.1', 'Functional_batch_norm_0_forward.output.1',
                  'torch.float16',
                  'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0.02, 0, 0, 0, 0.12, 0, 1, 1, 0.1, 1, 1, 1,
                  'Warning', '']
summary_line_3 = ['Functional_batch_norm_0_forward.output.2', 'Functional_batch_norm_0_forward.output.2',
                  'torch.float16',
                  'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0, 0, 0, 0, 2, 0, 1, 1, 1, 1, 1, 1,
                  'Warning', '']
line_input = ['Functional_batch_norm_0_forward.input.0', 'Functional_batch_norm_0_forward.input.0', 'torch.float16',
              'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 1, 1, 1, 0.95, 1, 1, 1, 1, 1, 1.01, 1, 1, 1,
              'Yes', '']
line_1 = ['Functional_batch_norm_0_forward.output.0', 'Functional_batch_norm_0_forward.output.0', 'torch.float16',
          'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0.8, 1, 1, 0.59, 1, 'nan', 0, 1, 1, 19, 1, 1, 1,
          'Warning', '']
line_2 = ['Functional_batch_norm_0_forward.output.1', 'Functional_batch_norm_0_forward.output.1', 'torch.float16',
          'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0.9, 1, 1, 0.8, 1, 0, 0.12, 0, 1, 1, 0.1, 1, 1, 1,
          'Warning', '']
line_3 = ['Functional_batch_norm_0_forward.output.2', 'Functional_batch_norm_0_forward.output.2', 'torch.float16',
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

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_data')
base_dir2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_data2')
base_dir3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_data3')
pt_dir = os.path.join(base_dir3, f'dump_data_dir')


def generate_dump_json(base_dir):
    data_path = os.path.join(base_dir, 'dump.json')
    data = {
        'task': 'statistics',
        'level': 'L1',
        'dump_data_dir': '',
        'data': {
            'Functional.linear.0.forward': {
                'input_args': [
                    {'type': 'torch.Tensor',
                     'dtype': 'torch.float32',
                     'shape': [2, 2],
                     'Max': 2,
                     'Min': 0,
                     'Mean': 1,
                     'Norm': 1,
                     'requires_grad': False,
                     'data_name': 'Functional.linear.0.forward.input.0.pt'
                     }
                ]
            }
        }
    }
    with open(data_path, 'w') as json_file:
        json.dump(data, json_file)


def generate_stack_json(base_dir):
    data_path = os.path.join(base_dir, 'stack.json')
    data = {'Functional.linear.0.forward': ['File']}
    with open(data_path, 'w') as json_file:
        json.dump(data, json_file)


def generate_pt(base_dir):
    data_path = os.path.join(base_dir, 'Functional.linear.0.forward.input.0.pt')
    data = torch.Tensor([1, 2, 3, 4])
    torch.save(data, data_path)


class TestUtilsMethods(unittest.TestCase):

    def setUp(self):
        os.makedirs(base_dir, mode=0o750, exist_ok=True)
        os.makedirs(base_dir2, mode=0o750, exist_ok=True)
        os.makedirs(base_dir3, mode=0o750, exist_ok=True)
        os.makedirs(pt_dir, mode=0o750, exist_ok=True)

        self.lock = threading.Lock()

    def tearDown(self):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        if os.path.exists(base_dir2):
            shutil.rmtree(base_dir2)
        if os.path.exists(pt_dir):
            shutil.rmtree(pt_dir)
        if os.path.exists(base_dir3):
            shutil.rmtree(base_dir3)

    def test_get_accuracy_graph_mode(self):
        result = []
        get_accuracy(result, npu_dict_aten, bench_dict_functional, highlight_dict)
        self.assertEqual(result, aten_result)

    def test_find_error_rows(self):
        summary_result = [summary_line_input, summary_line_1, summary_line_2, summary_line_3]
        highlight_dict = {'red_rows': [], 'yellow_rows': []}
        find_error_rows(summary_result, 0, 1, highlight_dict, dump_mode=Const.SUMMARY)
        self.assertEqual(highlight_dict, {'red_rows': [], 'yellow_rows': []})

    def test_find_compare_result_error_rows(self):
        result = [line_input, line_1, line_2, line_3]
        result_df = pd.DataFrame(result)
        highlight_dict = {'red_rows': [], 'yellow_rows': []}
        find_compare_result_error_rows(result_df, highlight_dict, dump_mode=Const.ALL)
        self.assertEqual(highlight_dict, {'red_rows': [num_1, num_3], 'yellow_rows': [num_2]})

    def test_calculate_summary_data(self):
        npu_summary_data = [1, 1, 1, 1]
        bench_summary_data = [2, 2, 2, 2]
        result_item = ['', '', '', '', '', '', '', '', '', '', '', '', '', '']
        Comparator().calculate_summary_data(npu_summary_data, bench_summary_data, result_item)
        self.assertEqual(result_item, ['', '', '', '', '', '', -1, -1, -1, -1, '50.0%', '50.0%', '50.0%', '50.0%', '', ''])

        bench_summary_data = [0, 0, 0, 0]
        result_item = ['', '', '', '', '', '', '', '', '', '', '', '', '', '']
        Comparator().calculate_summary_data(npu_summary_data, bench_summary_data, result_item)
        self.assertEqual(result_item, ['', '', '', '', '', '', 1, 1, 1, 1, 'N/A', 'N/A', 'N/A', 'N/A', 'Warning', 'Need double check api accuracy.'])

    def test_make_result_table_stack_mode_True(self):
        result_md5 = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                       'torch.float32', 'torch.float32', [2, 2], [2, 2], '', '', '', 'File']]
        result_summary = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                           'torch.float32', 'torch.float32', [2, 2], [2, 2], '', '', '', '', '', '', '', '',
                           1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', 'File']]
        result_all = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                       'torch.float32', 'torch.float32', [2, 2], [2, 2], '', '', '', '', '',
                       1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', 'File', '-1']]
        columns_md5_stack_mode_true = CompareConst.MD5_COMPARE_RESULT_HEADER + ['NPU_Stack_Info']
        result_table_md5_true = pd.DataFrame(result_md5, columns=columns_md5_stack_mode_true, dtype=object)
        columns_summary_stack_mode_true = CompareConst.SUMMARY_COMPARE_RESULT_HEADER + ['NPU_Stack_Info']
        result_table_summary_true = pd.DataFrame(result_summary, columns=columns_summary_stack_mode_true, dtype=object)
        columns_all_stack_mode_true = CompareConst.COMPARE_RESULT_HEADER + ['NPU_Stack_Info'] + ['Data_name']
        result_table_all_true = pd.DataFrame(result_all, columns=columns_all_stack_mode_true, dtype=object)

        stack_mode = True

        md5_compare = True
        summary_mode = False
        result_df = Comparator().make_result_table(result_md5, md5_compare, summary_mode, stack_mode)
        self.assertTrue(result_df.equals(result_table_md5_true))

        md5_compare = False
        summary_mode = True
        result_df = Comparator().make_result_table(result_summary, md5_compare, summary_mode, stack_mode)
        self.assertTrue(result_df.equals(result_table_summary_true))

        md5_compare = False
        summary_mode = False
        result_df = Comparator().make_result_table(result_all, md5_compare, summary_mode, stack_mode)
        self.assertTrue(result_df.equals(result_table_all_true))

    def test_make_result_table_stack_mode_False(self):
        result_md5_test = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                       'torch.float32', 'torch.float32', [2, 2], [2, 2], '', '', '', '']]
        result_md5 = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                       'torch.float32', 'torch.float32', [2, 2], [2, 2], '', '', '']]
        result_summary_test = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                           'torch.float32', 'torch.float32', [2, 2], [2, 2], '', '', '', '', '', '', '', '',
                           1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '']]
        result_summary = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                           'torch.float32', 'torch.float32', [2, 2], [2, 2], '', '', '', '', '', '', '', '',
                           1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '']]
        result_all_test = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                       'torch.float32', 'torch.float32', [2, 2], [2, 2], '', '', '', '', '',
                       1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '', '-1']]
        result_all = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                       'torch.float32', 'torch.float32', [2, 2], [2, 2], '', '', '', '', '',
                       1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '-1']]
        columns_md5_stack_mode_true = CompareConst.MD5_COMPARE_RESULT_HEADER
        result_table_md5_true = pd.DataFrame(result_md5, columns=columns_md5_stack_mode_true, dtype='object')
        columns_summary_stack_mode_true = CompareConst.SUMMARY_COMPARE_RESULT_HEADER
        result_table_summary_true = pd.DataFrame(result_summary, columns=columns_summary_stack_mode_true,
                                                 dtype='object')
        columns_all_stack_mode_true = CompareConst.COMPARE_RESULT_HEADER + ['Data_name']
        result_table_all_true = pd.DataFrame(result_all, columns=columns_all_stack_mode_true, dtype='object')

        stack_mode = False

        md5_compare = True
        summary_mode = False
        result_df = Comparator().make_result_table(result_md5_test, md5_compare, summary_mode, stack_mode)
        self.assertTrue(result_df.equals(result_table_md5_true))

        md5_compare = False
        summary_mode = True
        result_df = Comparator().make_result_table(result_summary_test, md5_compare, summary_mode, stack_mode)
        self.assertTrue(result_df.equals(result_table_summary_true))

        md5_compare = False
        summary_mode = False
        result_df = Comparator().make_result_table(result_all_test, md5_compare, summary_mode, stack_mode)
        self.assertTrue(result_df.equals(result_table_all_true))

    def test_gen_merge_list(self):
        dump_mode = Const.SUMMARY
        op_data = {
            'input_args': [
                {
                    'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [2, 2],
                    'Max': 1, 'Min': 1, 'Mean': 1, 'Norm': 1, 'requires_grad': False,
                    'data_name': 'Functional.linear.0.forward.input.0.pt',
                    'full_op_name': 'Functional.linear.0.forward.input.0'
                }
            ]
        }
        json_data = {'data': {'Functional.linear.0.forward': op_data}}
        op_name = 'Functional.linear.0.forward'
        stack_json_data = {'Functional.linear.0.forward': ['File']}
        merge_list = {
            'input_struct': [('torch.float32', [2, 2])],
            'op_name': ['Functional.linear.0.forward.input.0'],
            'output_struct': [],
            'stack_info': [['File']],
            'summary': [[1, 1, 1, 1]]
        }
        result = Comparator().gen_merge_list(json_data, op_name, stack_json_data, dump_mode)
        self.assertEqual(result, merge_list)

    def test_check_op_fuzzy_false(self):
        fuzzy_match = False
        pt_comparator = PTComparator()
        result = pt_comparator.check_op(npu_dict, bench_dict, fuzzy_match)
        self.assertEqual(result, True)

    def test_check_op_fuzzy_true(self):
        fuzzy_match = True
        pt_comparator = PTComparator()
        result = pt_comparator.check_op(npu_dict2, bench_dict, fuzzy_match)
        self.assertEqual(result, True)

    def test_match_op_both_last_element(self):
        fuzzy_match = False
        pt_comparator = PTComparator()
        a, b = pt_comparator.match_op([npu_dict], [bench_dict], fuzzy_match)
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_match_op_only_npu_last_element(self):
        fuzzy_match = False
        pt_comparator = PTComparator()
        a, b = pt_comparator.match_op([npu_dict], [bench_dict, 1], fuzzy_match)
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_match_op_only_bench_last_element(self):
        fuzzy_match = False
        pt_comparator = PTComparator()
        a, b = pt_comparator.match_op([npu_dict, npu_dict2], [bench_dict], fuzzy_match)
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_compare_process(self):
        generate_dump_json(base_dir)
        generate_stack_json(base_dir)
        file_lists = [os.path.join(base_dir, 'dump.json'), os.path.join(base_dir, 'dump.json'), os.path.join(base_dir, 'stack.json')]
        stack_mode = True
        fuzzy_match = False
        dump_mode = Const.SUMMARY
        result = PTComparator().compare_process(file_lists, stack_mode, fuzzy_match, dump_mode)
        o_data = [
            ['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
             'torch.float32', 'torch.float32', [2, 2], [2, 2], 0, 0, 0, 0, '0.0%', 'N/A', '0.0%', '0.0%',
             2, 0, 1, 1, 2, 0, 1, 1, '', '', ['File']
             ]
        ]
        columns = CompareConst.SUMMARY_COMPARE_RESULT_HEADER + ['NPU_Stack_Info']
        o_result = pd.DataFrame(o_data, columns=columns, dtype=object)
        self.assertTrue(result.equals(o_result))

    def test_merge_data(self):
        dump_mode = Const.SUMMARY
        op_data = {
            'input_args': [
                {
                    'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [2, 2],
                    'Max': 1, 'Min': 1, 'Mean': 1, 'Norm': 1, 'requires_grad': False,
                    'data_name': 'Functional.linear.0.forward.input.0.pt',
                    'full_op_name': 'Functional.linear.0.forward.input.0'
                }
            ]
        }
        json_data = {'data': {'Functional.linear.0.forward': op_data}}
        stack_json_data = {'Functional.linear.0.forward': ['File']}
        result = Comparator().merge_data(json_data, stack_json_data, dump_mode)
        ops_all = {
            'Functional.linear.0.forward.input.0': {
                'data_name': None, 'stack_info': [['File']],
                'struct': ('torch.float32', [2, 2]), 'summary': [1, 1, 1, 1]
            }
        }
        self.assertEqual(result, ops_all)

    def test_compare_core_basic(self):
        generate_dump_json(base_dir2)
        generate_stack_json(base_dir2)
        input_params = {
            "npu_json_path": os.path.join(base_dir2, "dump.json"),
            "bench_json_path": os.path.join(base_dir2, "dump.json"),
            "stack_json_path": os.path.join(base_dir2, "stack.json"),
        }
        output_path = base_dir2

        PTComparator().compare_core(input_params, output_path, summary_compare=True)

        output_files = os.listdir(output_path)
        self.assertTrue(any(f.endswith(".xlsx") for f in output_files))

    def test_compare_ops(self):
        generate_dump_json(base_dir3)
        generate_stack_json(base_dir3)
        generate_pt(pt_dir)
        dump_path = os.path.join(base_dir3, 'dump.json')
        stack_path = os.path.join(base_dir3, 'stack.json')
        input_param = {'npu_json_path': dump_path, 'bench_json_path': dump_path, 'stack_json_path': stack_path,
                       'is_print_compare_log': True, 'npu_dump_data_dir': pt_dir, 'bench_dump_data_dir': pt_dir}
        dump_path_dict = {'Functional.linear.0.forward.input.0': ['Functional.linear.0.forward.input.0.pt',
                                                                  'Functional.linear.0.forward.input.0.pt']}
        result_df = pd.DataFrame({
            'NPU Name': ['Functional.linear.0.forward.input.0'],
            'Bench Name': ['Functional.linear.0.forward.input.0']
        })
        updated_df = PTComparator().compare_ops(idx=0, dump_path_dict=dump_path_dict, result_df=result_df, lock=self.lock,
                                              input_param=input_param)

        self.assertEqual(updated_df.loc[0, CompareConst.COSINE], 1.0)
        self.assertEqual(updated_df.loc[0, CompareConst.MAX_ABS_ERR], 0)

    def test_do_multi_process(self):
        data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                 'torch.float32', 'torch.float32', [2, 2], [2, 2],
                 '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, 'Yes', '', '-1']]
        o_data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                   'torch.float32', 'torch.float32', [2, 2], [2, 2], 'None', 'None', 'None', 'None', 'None',
                   1, 1, 1, 1, 1, 1, 1, 1, 'None', 'No bench data matched.', '-1']]
        columns = CompareConst.COMPARE_RESULT_HEADER + ['Data_name']
        result_df = pd.DataFrame(data, columns=columns)
        o_result = pd.DataFrame(o_data, columns=columns)
        input_param = {}
        result = Comparator()._do_multi_process(input_param, result_df)
        self.assertTrue(result.equals(o_result))

    def test_compare_by_op_1(self):
        npu_op_name = 'Functional.linear.0.forward.input.0'
        bench_op_name = 'N/A'
        op_name_mapping_dict = {'Functional.linear.0.forward.input.0': [-1, -1]}
        input_param = {}
        result = PTComparator().compare_by_op(npu_op_name, bench_op_name, op_name_mapping_dict, input_param)
        self.assertEqual(result, ['None', 'None', 'None', 'None', 'None', 'No bench data matched.'])

    def test_compare_by_op_2(self):
        npu_op_name = 'Functional.linear.0.forward.input.0'
        bench_op_name = 'Functional.linear.0.forward.input.0'
        pt_name = 'Functional.linear.0.forward.input.0.pt'
        pt_path = os.path.join(base_dir, pt_name)
        op_name_mapping_dict = {'Functional.linear.0.forward.input.0': [pt_path, pt_path]}
        input_param = {'npu_dump_data_dir': base_dir, 'bench_dump_data_dir': base_dir}
        result = PTComparator().compare_by_op(npu_op_name, bench_op_name, op_name_mapping_dict, input_param)
        self.assertEqual(result, ['None', 'None', 'None', 'None', 'None', f'Dump file: {pt_path} not found.'])

        generate_pt(base_dir)
        result = PTComparator().compare_by_op(npu_op_name, bench_op_name, op_name_mapping_dict, input_param)
        self.assertEqual(result, [1.0, 0.0, 0.0, 1.0, 1.0, ''])
