# coding=utf-8
import json
import os
import shutil
import threading
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from msprobe.core.common.const import CompareConst, Const
from msprobe.core.common.utils import CompareException
from msprobe.core.compare.acc_compare import Comparator
from msprobe.core.compare.highlight import ApiBatch
from msprobe.core.compare.utils import get_accuracy


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
     'Min': -0.331031858921051, 'Mean': -0.030964046716690063, 'Norm': 2.2533628940582275, 'requires_grad': True,
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
                              [0.003992878366261721, -0.008102823048830032, -0.0002002553956117481,
                               0.02844562754034996],
                              [-0.1, -0.1, -0.1, -0.1],
                              [0.33033010363578796, -0.331031858921051, -0.030964046716690063, 2.2533628940582275]],
                  'stack_info': []}

o_result = [
    ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.0', 'torch.float32', 'torch.float32',
     [1, 1, 28, 28], [1, 1, 28, 28], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 3.029174327850342,
     -2.926689624786377,
     -0.06619918346405029, 3.029174327850342, -2.926689624786377, -0.06619918346405029, '', '', 'None'],
    ['Functional.conv2d.0.forward.input.1', 'Functional.conv2d.0.forward.input.1', 'torch.float32', 'torch.float32',
     [16, 1, 5, 5], [16, 1, 5, 5], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 0.19919930398464203,
     -0.19974489510059357,
     0.006269412115216255, 0.19919930398464203, -0.19974489510059357, 0.006269412115216255, '', '', 'None'],
    ['Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.input.2', 'torch.float32', 'torch.float32',
     [16], [16], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 0.19734230637550354, -0.18177609145641327,
     0.007903944700956345,
     0.19734230637550354, -0.18177609145641327, 0.007903944700956345, '', '', 'None'],
    ['Functional.conv2d.0.forward.output', 'Functional.conv2d.0.forward.output', 'torch.float32', 'torch.float32',
     [1, 16, 28, 28], [1, 16, 28, 28], 0.0, 0.0, 0.0, ' ', '0.0%', '0.0%', '0.0%', ' ', 2.1166646480560303,
     -2.190781354904175,
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
     ' ', ' ', ' ', ' ', ' ', ' ', 0.30550330877304077, -0.24485322833061218, -0.010361209511756897, 'Nan', 'Nan', 'Nan',
     'Yes', '', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.2', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', ' ', 623.9192504882812, 432.96826171875, 520.2276611328125, 'Nan', 'Nan', 'Nan',
     'Yes', '', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.3', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', ' ', 2.4797861576080322, -3.055997371673584, -0.04795549064874649, 'Nan', 'Nan', 'Nan',
     'Yes', '', 'None'],
    ['Aten__native_batch_norm_legit_functional.default_0_forward.output.4', 'Nan', 'torch.float32', 'Nan', [256], 'Nan',
     ' ', ' ', ' ', ' ', ' ', ' ', 61.7945556640625, 42.59713363647461, 52.03831481933594, 'Nan', 'Nan', 'Nan',
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
line_input = ['Functional.batch.norm.0.forward.input.0', 'Functional.batch.norm.0.forward.input.0', 'torch.float16',
              'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 1, 0.5, 1, 1, 0.95, 1,
              1, 1, 1, 1, 1.01, 1, 1, 1,
              'Yes', '']
line_1 = ['Functional.batch.norm.0.forward.output.0', 'Functional.batch.norm.0.forward.output.0', 'torch.float16',
          'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0.8, 0.5, 1, 1, 0.59, 1,
          'nan', 0, 1, 1, 19, 1, 1, 1,
          'Yes', '']
line_2 = ['Functional.batch.norm.0.forward.output.1', 'Functional.batch.norm.0.forward.output.1', 'torch.float16',
          'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0.9, 0.5, 1, 1, 0.8, 1,
          0, 0.12, 0, 1, 1, 0.1, 1, 1,
          'Yes', '']
line_3 = ['Functional.batch.norm.0.forward.output.2', 'Functional.batch.norm.0.forward.output.2', 'torch.float16',
          'torch.float32', [256, 256, 14, 14], [256, 256, 14, 14], 0.8, 0.5, 1.1e+10, 1, 0.85, 1,
          9, 0.12, 0, 1, 1, 0.1, 1, 1,
          'Yes', '']

op_data = {
    'input_args': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                    'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
                    'Norm': 2.2533628940582275, 'requires_grad': True},
                   {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                    'Max': 0.003992878366261721, 'Min': -0.008102823048830032, 'Mean': -0.0002002553956117481,
                    'Norm': 0.02844562754034996, 'requires_grad': False}],
    'input_kwargs': {'alpha': {'type': 'float', 'value': -0.1}},
    'output': [{'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [16, 1, 3, 3],
                'Max': 0.33033010363578796, 'Min': -0.331031858921051, 'Mean': -0.030964046716690063,
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
        get_accuracy(result, npu_dict_aten, bench_dict_functional, dump_mode=Const.SUMMARY)
        self.assertEqual(result, aten_result)
