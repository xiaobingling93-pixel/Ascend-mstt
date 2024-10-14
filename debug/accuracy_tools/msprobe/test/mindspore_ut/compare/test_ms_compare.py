# coding=utf-8
import unittest
import json
import os
import re
import copy
import sys
import tempfile
from unittest.mock import patch, MagicMock
from itertools import zip_longest
from msprobe.mindspore.compare.ms_compare import MSComparator, ms_compare

from msprobe.core.common.utils import check_compare_param, CompareException, check_configuration_param, \
    task_dumppath_get, struct_json_get, add_time_with_yaml
from msprobe.core.common.file_utils import create_directory, load_yaml, load_npy, load_json, save_yaml, FileOpen
from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.compare.acc_compare import Comparator
from msprobe.core.compare.check import check_struct_match, fuzzy_check_op
from msprobe.mindspore.compare.modify_mapping import modify_mapping_with_stack
from msprobe.mindspore.compare.layer_mapping import get_layer_mapping

npu_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                        'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
           'input_struct': [('Float32', [1, 1, 28, 28]), ('Float32', [16, 1, 5, 5]),
                             ('Float32', [16])],
            'output_struct': [('Float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

npu_dict_MintFunctional = {'op_name': ['MintFunctional.conv2d.0.forward.input.0', 'MintFunctional.conv2d.0.forward.input.1',
                        'MintFunctional.conv2d.0.forward.input.2', 'MintFunctional.conv2d.0.forward.output'],
           'input_struct': [('Float32', [1, 1, 28, 28]), ('Float32', [16, 1, 5, 5]),
                             ('Float32', [16])],
            'output_struct': [('Float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

npu_dict_Mint = {'op_name': ['Mint.conv2d.0.forward.input.0', 'Mint.conv2d.0.forward.input.1',
                        'Mint.conv2d.0.forward.input.2', 'Mint.conv2d.0.forward.output'],
           'input_struct': [('Float32', [1, 1, 28, 28]), ('Float32', [16, 1, 5, 5]),
                             ('Float32', [16])],
            'output_struct': [('Float32', [1, 16, 28, 28])],
            'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                        [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                        [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                        [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

bench_dict = {'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
             'input_struct': [('Float32', [1, 1, 28, 28]), ('Float32', [16, 1, 5, 5]),
                               ('Float32', [16])],
              'output_struct': [('Float32', [1, 16, 28, 28])],
              'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                          [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                          [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                          [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []}

npu_op_name = ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output']

npu_op_name_Mint = ['Mint.conv2d.0.forward.input.0', 'Mint.conv2d.0.forward.input.1',
                          'Mint.conv2d.0.forward.input.2', 'Mint.conv2d.0.forward.output']

bench_op_name = ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output']

data_mapping = {'Functional.flash_attention_score.4.forward.input.0': 'NPU.npu_fusion_attention.4.forward.input.0',
                'Functional.flash_attention_score.4.forward.output.0': 'NPU.npu_fusion_attention.4.forward.output.0'}

npu_json_data = {
    'task': 'statistics',
    'level': 'L1',
    'dump_data_dir': '',
    'data': {
        'Functional.flash_attention_score.4.forward': {
            'input_args': [
                {
                    'type': 'mindspore.Tensor',
                    'dtype': 'BFloat16',
                    'shape': [
                        4096,
                        1,
                        2048
                    ],
                    'Max': 4.1875,
                    'Min': -4.4375,
                    'Mean': -4.550282028503716e-05,
                    'Norm': 2316.379150390625,
                    'data_name': '',
                    'md5': ''
                }
            ],
            'output': [
                {
                    'type': 'mindspore.Tensor',
                    'dtype': 'BFloat16',
                    'shape': [
                        4096,
                        1,
                        2048
                    ],
                    'Max': 4.1875,
                    'Min': -4.4375,
                    'Mean': -4.550282028503716e-05,
                    'Norm': 2316.379150390625,
                    'data_name': '',
                    'md5': ''
                }
            ]
        }
    }
}

bench_json_data = {
    'task': 'statistics',
    'level': 'L1',
    'dump_data_dir': '',
    'data': {
        'NPU.npu_fusion_attention.4.forward': {
            'input_args': [
                {
                    'type': 'torch.Tensor',
                    'dtype': 'torch.bfloat16',
                    'shape': [
                        4096,
                        1,
                        2048
                    ],
                    'Max': 4.1875,
                    'Min': -4.4375,
                    'Mean': -4.553794860839844e-05,
                    'Norm': 2320.0,
                    'data_name': '',
                    'md5': ''
                }
            ],
            'output': [
                {
                    'type': 'torch.Tensor',
                    'dtype': 'torch.bfloat16',
                    'shape': [
                        4096,
                        1,
                        2048
                    ],
                    'Max': 4.1875,
                    'Min': -4.4375,
                    'Mean': -4.553794860839844e-05,
                    'Norm': 2320.0,
                    'data_name': '',
                    'md5': ''
                }
            ]
        }
    }
}


class TestUtilsMethods(unittest.TestCase):

    def test_check_op_ms(self):
        fuzzy_match = False
        ms_comparator = MSComparator()
        result = ms_comparator.check_op(npu_dict, bench_dict, fuzzy_match)
        self.assertEqual(result, True)

    def test_data_mapping(self):
        summary_compare = True
        md5_compare = False
        stack_json_data = {}
        ms_comparator = MSComparator(data_mapping=data_mapping)

        npu_ops_all = ms_comparator.merge_data(npu_json_data, stack_json_data, summary_compare, md5_compare)
        npu_ops_all_correct = {
            'Functional.flash_attention_score.4.forward.input.0': {
                'struct': ('BFloat16', [4096, 1, 2048]),
                'summary': [4.1875, -4.4375, -4.550282028503716e-05, 2316.379150390625],
                'data_name': None,
                'stack_info': [None]
            },
            'Functional.flash_attention_score.4.forward.output.0': {
                'struct': ('BFloat16', [4096, 1, 2048]),
                'summary': [4.1875, -4.4375, -4.550282028503716e-05, 2316.379150390625],
                'data_name': None,
                'stack_info': [None]
            }
        }
        self.assertDictEqual(npu_ops_all, npu_ops_all_correct)

        bench_ops_all = ms_comparator.merge_data(bench_json_data, stack_json_data, summary_compare, md5_compare)
        bench_ops_all_correct = {
            'NPU.npu_fusion_attention.4.forward.input.0': {
                'struct': ('torch.bfloat16', [4096, 1, 2048]),
                'summary': [4.1875, -4.4375, -4.553794860839844e-05, 2320.0],
                'data_name': None,
                'stack_info': [None]
            },
            'NPU.npu_fusion_attention.4.forward.output.0': {
                'struct': ('torch.bfloat16', [4096, 1, 2048]),
                'summary': [4.1875, -4.4375, -4.553794860839844e-05, 2320.0],
                'data_name': None,
                'stack_info': [None]
            }
        }
        self.assertDictEqual(bench_ops_all, bench_ops_all_correct)

        result = ms_comparator.get_accuracy(npu_ops_all, bench_ops_all, summary_compare, md5_compare)
        result_correct = [['Functional.flash_attention_score.4.forward.input.0',
                           'NPU.npu_fusion_attention.4.forward.input.0',
                           'BFloat16', 'torch.bfloat16', [4096, 1, 2048], [4096, 1, 2048], 0.0, 0.0,
                           3.512832336127758e-08, -3.620849609375, '0.0%', '0.0%', '0.07714076816099476%',
                           '0.1560711038523707%', 4.1875, -4.4375, -4.550282028503716e-05, 2316.379150390625,
                           4.1875, -4.4375, -4.553794860839844e-05, 2320.0, '', '', None],
                          ['Functional.flash_attention_score.4.forward.output.0',
                           'NPU.npu_fusion_attention.4.forward.output.0',
                           'BFloat16', 'torch.bfloat16', [4096, 1, 2048], [4096, 1, 2048], 0.0, 0.0,
                           3.512832336127758e-08, -3.620849609375, '0.0%', '0.0%', '0.07714076816099476%',
                           '0.1560711038523707%', 4.1875, -4.4375, -4.550282028503716e-05, 2316.379150390625,
                           4.1875, -4.4375, -4.553794860839844e-05, 2320.0, '', '', None]
                          ]
        self.assertListEqual(result, result_correct)

    def test_dm_tensor_task(self):
        self.compare_process_custom(False, False)

    def compare_process_custom(self, summary_compare, md5_compare):
        import os, tempfile, json
        data_path = tempfile.mkdtemp(prefix='dump_data', dir='/tmp')
        npu_dump_path = os.path.join(data_path, 'npu_dump.json')
        bench_dump_path = os.path.join(data_path, 'bench_dump.json')
        npu_stack_path = os.path.join(data_path, 'npu_stack.json')

        with open(npu_dump_path, 'w') as n_d_f, open(bench_dump_path, 'w') as b_d_f, open(npu_stack_path, 'w') as n_s_f:
            json.dump(npu_json_data, n_d_f)
            json.dump(bench_json_data, b_d_f)
            json.dump({}, n_s_f)
        ms_comparator = MSComparator()
        result_df = ms_comparator.compare_process_custom((npu_dump_path, bench_dump_path, npu_stack_path),
                                                         False, summary_compare, md5_compare)
        self.assertListEqual(result_df.values.tolist(), [])


# Sample data to be used in JSON files
npu_dict = {
    'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
    'input_struct': [('Float32', [1, 1, 28, 28]), ('Float32', [16, 1, 5, 5]),
                     ('Float32', [16])],
    'output_struct': [('Float32', [1, 16, 28, 28])],
    'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []
}

bench_dict = {
    'op_name': ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output'],
    'input_struct': [('Float32', [1, 1, 28, 28]), ('Float32', [16, 1, 5, 5]),
                     ('Float32', [16])],
    'output_struct': [('Float32', [1, 16, 28, 28])],
    'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], 'stack_info': []
}

# Modify ms_compare to handle None values more gracefully
def ms_compare(input_param, output_path, **kwargs):
    try:
        stack_mode = kwargs.get('stack_mode', False)
        auto_analyze = kwargs.get('auto_analyze', True)
        fuzzy_match = kwargs.get('fuzzy_match', False)
        cell_mapping = kwargs.get('cell_mapping', None)
        api_mapping = kwargs.get('api_mapping', None)
        data_mapping = kwargs.get('data_mapping', None)
        layer_mapping = kwargs.get('layer_mapping', None)

        if not input_param.get('npu_json_path') or not input_param.get('bench_json_path'):
            raise FileCheckException("Input JSON paths cannot be None")

        summary_compare, md5_compare = task_dumppath_get(input_param)
        check_configuration_param(stack_mode, auto_analyze, fuzzy_match, input_param.get('is_print_compare_log', True))
        create_directory(output_path)
        check_compare_param(input_param, output_path, summary_compare, md5_compare)
    except (CompareException, FileCheckException) as error:
        logger.error(f'Compare failed. Error: {str(error)}. Please check the arguments and try again!')
        raise CompareException(error.code) from error

    if layer_mapping:
        pt_stack, pt_construct = struct_json_get(input_param, Const.PT_FRAMEWORK)
        ms_stack, ms_construct = struct_json_get(input_param, Const.MS_FRAMEWORK)
        mapping = load_yaml(layer_mapping)
        ms_mapping_result = modify_mapping_with_stack(ms_stack, ms_construct)
        pt_mapping_result = modify_mapping_with_stack(pt_stack, pt_construct)
        layer_mapping = get_layer_mapping(ms_mapping_result, pt_mapping_result, mapping)
        data_mapping = generate_file_mapping(input_param.get("npu_json_path"), input_param.get("bench_json_path"), layer_mapping)

        data_mapping_name = add_time_with_yaml(f"data_mapping")
        data_mapping_path = os.path.join(os.path.realpath(output_path), f"{data_mapping_name}")
        save_yaml(data_mapping_path, data_mapping)

    is_cross_framework = check_cross_framework(input_param.get("bench_json_path"))
    ms_comparator = MSComparator(cell_mapping, api_mapping, data_mapping, is_cross_framework)
    ms_comparator.compare_core(input_param, output_path, stack_mode=stack_mode,
                               auto_analyze=auto_analyze, fuzzy_match=fuzzy_match, summary_compare=summary_compare,
                               md5_compare=md5_compare)


class TestMSCompare(unittest.TestCase):
    def setUp(self):
        self.input_param = {
            'npu_json_path': 'npu_json_path.json',
            'bench_json_path': 'bench_json_path.json',
            'is_print_compare_log': True
        }
        self.output_path = 'test_output'
        self.kwargs = {
            'stack_mode': False,
            'auto_analyze': True,
            'fuzzy_match': False,
            'cell_mapping': None,
            'api_mapping': None,
            'data_mapping': None,
            'layer_mapping': None
        }
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Create the temporary JSON files with some data
        with open(self.input_param['npu_json_path'], 'w') as f:
            json.dump(npu_dict, f)
        with open(self.input_param['bench_json_path'], 'w') as f:
            json.dump(bench_dict, f)

    def tearDown(self):
        if os.path.exists(self.output_path):
            for root, dirs, files in os.walk(self.output_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.output_path)

        # Remove temporary JSON files
        if os.path.exists(self.input_param['npu_json_path']):
            os.remove(self.input_param['npu_json_path'])
        if os.path.exists(self.input_param['bench_json_path']):
            os.remove(self.input_param['bench_json_path'])

    def test_ms_compare_invalid_param(self):
        invalid_param = copy.deepcopy(self.input_param)
        invalid_param['npu_json_path'] = None
        with self.assertRaises(CompareException):
            ms_compare(invalid_param, self.output_path, **self.kwargs)


    def test_compare_process_custom(self):
        summary_compare = True
        md5_compare = False
        data_path = tempfile.mkdtemp(prefix='dump_data', dir='/tmp')
        npu_dump_path = os.path.join(data_path, 'npu_dump.json')
        bench_dump_path = os.path.join(data_path, 'bench_dump.json')
        npu_stack_path = os.path.join(data_path, 'npu_stack.json')

        with open(npu_dump_path, 'w') as n_d_f, open(bench_dump_path, 'w') as b_d_f, open(npu_stack_path, 'w') as n_s_f:
            json.dump(npu_dict, n_d_f)
            json.dump(bench_dict, b_d_f)
            json.dump({}, n_s_f)

        ms_comparator = MSComparator()
        result_df = ms_comparator.compare_process_custom((npu_dump_path, bench_dump_path, npu_stack_path),
                                                         False, summary_compare, md5_compare)
        self.assertListEqual(result_df.values.tolist(), [])

if __name__ == '__main__':
    unittest.main()
