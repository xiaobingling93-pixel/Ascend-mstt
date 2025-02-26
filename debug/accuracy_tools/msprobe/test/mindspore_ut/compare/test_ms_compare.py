# coding=utf-8
import json
import os
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from msprobe.core.common.utils import CompareException
from msprobe.core.compare.acc_compare import ModeConfig
from msprobe.mindspore.compare.ms_compare import MappingConfig, MSComparator, check_cross_framework
from msprobe.core.common.const import Const

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

npu_op_name_list = ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output']

npu_op_name_Mint = ['Mint.conv2d.0.forward.input.0', 'Mint.conv2d.0.forward.input.1',
                          'Mint.conv2d.0.forward.input.2', 'Mint.conv2d.0.forward.output']

bench_op_name = ['Functional.conv2d.0.forward.input.0', 'Functional.conv2d.0.forward.input.1',
                          'Functional.conv2d.0.forward.input.2', 'Functional.conv2d.0.forward.output']

data_mapping = {'Functional.flash_attention_score.4.forward.input.0': 'NPU.npu_fusion_attention.4.forward.input.0',
                'Functional.flash_attention_score.4.forward.output.0': 'NPU.npu_fusion_attention.4.forward.output.0'}

npu_cell_dict = {'op_name': ['Cell.fc1.Dense.forward.0.input.0', 'Cell.fc1.Dense.forward.0.input.1',
                             'Cell.fc1.Dense.forward.0.input.2', 'Cell.fc1.Dense.forward.0.output.0'],
                 'input_struct': [('Float32', [1, 1, 28, 28]), ('Float32', [16, 1, 5, 5]),
                                  ('Float32', [16])],
                 'output_struct': [('Float32', [1, 16, 28, 28])],
                 'summary': [[3.029174327850342, -2.926689624786377, -0.06619918346405029],
                          [0.19919930398464203, -0.19974489510059357, 0.006269412115216255],
                          [0.19734230637550354, -0.18177609145641327, 0.007903944700956345],
                          [2.1166646480560303, -2.190781354904175, -0.003579073818400502]], "stack_info": []}

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


json_data_template = {
    'task': 'statistics',
    'level': 'L1',
    'dump_data_dir': '',
    'data': {}
}


def gen_data(is_ms=True):
    type_value = 'mindspore.Tensor' if is_ms else 'torch.Tensor'
    dtype_value = 'BFloat16' if is_ms else 'torch.bfloat16'
    return {
        'type': type_value,
        'dtype': dtype_value,
        'shape': [4096, 1, 2048],
        'Max': random.uniform(0, 4),
        'Min': random.uniform(-4, 0),
        'Mean': random.random() / 10000,
        'Norm': random.random() * 1000
    }


def gen_api_mapping_test_data(need_user_mapping=False):
    result_npu = json_data_template.copy()
    result_bench = json_data_template.copy()

    stack_mode = True
    auto_analyze = True
    fuzzy_match = False
    dump_mode = Const.SUMMARY

    mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)
    mapping_config = MappingConfig()
    ms_comparator = MSComparator(mode_config, mapping_config)

    api_mapping = ms_comparator.load_internal_api()
    ms_api_list = np.random.choice(list(api_mapping.keys()), size=5, replace=False).astype(str).tolist()
    ms_api_data = {}
    pt_api_data = {}
    user_mapping = []
    for api in ms_api_list:
        call_num = random.randint(1, 10)
        direction = random.choice(['forward', 'backward'])
        data_name_ms = api + '.' + str(call_num) + '.' + direction
        data_name_pt = api_mapping.get(api) + '.' + str(call_num) + '.' + direction
        input_num = random.randint(1, 5)
        output_num = random.randint(1, 5)
        ms_data = {'input_args': [gen_data(True) for _ in range(input_num)],
                   'output': [gen_data(True) for _ in range(output_num)]}
        pt_data = {'input_args': [gen_data(False) for _ in range(input_num)],
                   'output': [gen_data(False) for _ in range(output_num)]}
        ms_api_data[data_name_ms] = ms_data
        pt_api_data[data_name_pt] = pt_data
        if need_user_mapping:
            compare_num_input = random.randint(1, input_num)
            compare_num_output = random.randint(1, output_num)
            user_mapping_item = {'ms_api': api,
                                 'pt_api': api_mapping.get(api),
                                 'ms_args': sorted(np.random.choice(list(range(input_num)), size=compare_num_input,
                                                                    replace=False).astype(int).tolist()),
                                 'pt_args': sorted(np.random.choice(list(range(input_num)), size=compare_num_input,
                                                                    replace=False).astype(int).tolist()),
                                 'ms_output': sorted(np.random.choice(list(range(output_num)), size=compare_num_output,
                                                                    replace=False).astype(int).tolist()),
                                 'pt_output': sorted(np.random.choice(list(range(output_num)), size=compare_num_output,
                                                                    replace=False).astype(int).tolist())}
            user_mapping.append(user_mapping_item)
    ms_api_key_list = list(ms_api_data.keys())
    random.shuffle(ms_api_key_list)
    result_npu['data'] = {k: ms_api_data.get(k) for k in ms_api_key_list}
    pt_api_key_list = list(pt_api_data.keys())
    random.shuffle(pt_api_key_list)
    result_bench['data'] = {k: pt_api_data.get(k) for k in pt_api_key_list}
    return result_npu, result_bench, user_mapping


class TestUtilsMethods(unittest.TestCase):

    def test_check_op_ms(self):
        stack_mode = True
        auto_analyze = True
        fuzzy_match = False
        dump_mode = Const.ALL

        mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)
        mapping_config = MappingConfig()

        ms_comparator = MSComparator(mode_config, mapping_config)
        result = ms_comparator.check_op(npu_dict, bench_dict)
        self.assertTrue(result)

    def test_data_mapping(self):
        stack_json_data = {}

        stack_mode = True
        auto_analyze = True
        fuzzy_match = False
        dump_mode = Const.SUMMARY

        mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)
        mapping_config = MappingConfig(data_mapping=data_mapping)
        ms_comparator = MSComparator(mode_config, mapping_config)

        npu_ops_all = ms_comparator.merge_data(npu_json_data, stack_json_data)
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

        bench_ops_all = ms_comparator.merge_data(bench_json_data, stack_json_data)
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

        result = ms_comparator.get_accuracy(npu_ops_all, bench_ops_all)
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
        self.compare_process_custom(dump_mode=Const.ALL)

    def compare_process_custom(self, dump_mode):
        data_path = tempfile.mkdtemp(prefix='dump_data', dir='/tmp')
        try:
            npu_dump_path = os.path.join(data_path, 'npu_dump.json')
            bench_dump_path = os.path.join(data_path, 'bench_dump.json')
            npu_stack_path = os.path.join(data_path, 'npu_stack.json')

            with open(npu_dump_path, 'w') as n_d_f:
                json.dump(npu_json_data, n_d_f)
            with open(bench_dump_path, 'w') as b_d_f:
                json.dump(bench_json_data, b_d_f)
            with open(npu_stack_path, 'w') as n_s_f:
                json.dump({}, n_s_f)

            stack_mode = True
            auto_analyze = True
            fuzzy_match = False
            dump_mode = Const.SUMMARY

            mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)
            mapping_config = MappingConfig()

            ms_comparator = MSComparator(mode_config, mapping_config)
            result_df = ms_comparator.compare_process_custom((npu_dump_path, bench_dump_path, npu_stack_path))
            self.assertListEqual(result_df.values.tolist(), [])
        finally:
            shutil.rmtree(data_path)

    @patch('msprobe.mindspore.compare.ms_compare.detect_framework_by_dump_json')
    def test_check_cross_framework_valid_pytorch(self, mock_detect_framework):
        mock_detect_framework.return_value = Const.PT_FRAMEWORK

        result = check_cross_framework("dummy_path")

        self.assertTrue(result)

    @patch('msprobe.mindspore.compare.ms_compare.detect_framework_by_dump_json')
    def test_check_cross_framework_invalid_framework(self, mock_detect_framework):
        mock_detect_framework.return_value = Const.MS_FRAMEWORK

        result = check_cross_framework("dummy_path")

        self.assertFalse(result)

    def test_comapre_process(self):
        data_path = tempfile.mkdtemp(prefix='dump_data', dir='/tmp')
        try:
            npu_dump_path = os.path.join(data_path, 'npu_dump.json')
            bench_dump_path = os.path.join(data_path, 'bench_dump.json')
            npu_stack_path = os.path.join(data_path, 'npu_stack.json')

            npu_data, bench_data, _ = gen_api_mapping_test_data()
            with open(npu_dump_path, 'w', encoding='utf8') as n_d_f:
                json.dump(npu_data, n_d_f)
            with open(bench_dump_path, 'w', encoding='utf8') as b_d_f:
                json.dump(bench_data, b_d_f)
            with open(npu_stack_path, 'w', encoding='utf8') as n_s_f:
                json.dump({}, n_s_f)

            stack_mode = True
            auto_analyze = True
            fuzzy_match = False
            dump_mode = Const.SUMMARY
            mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)
            mapping_config = MappingConfig(api_mapping=True)

            ms_comparator = MSComparator(mode_config, mapping_config)
            result_df = ms_comparator.compare_process((npu_dump_path, bench_dump_path, npu_stack_path))
            self.assertTrue((result_df['Bench Name'] != 'N/A').all())
        finally:
            shutil.rmtree(data_path)
    
    def test_compare_process_with_customize_api_mapping(self):
        data_path = tempfile.mkdtemp(prefix='dump_data', dir='/tmp')
        try:
            npu_dump_path = os.path.join(data_path, 'npu_dump.json')
            bench_dump_path = os.path.join(data_path, 'bench_dump.json')
            npu_stack_path = os.path.join(data_path, 'npu_stack.json')
            user_mapping_path = os.path.join(data_path, 'user_mapping.yaml')

            npu_data, bench_data, user_mapping = gen_api_mapping_test_data(True)
            with open(npu_dump_path, 'w', encoding='utf8') as n_d_f:
                json.dump(npu_data, n_d_f)
            with open(bench_dump_path, 'w', encoding='utf8') as b_d_f:
                json.dump(bench_data, b_d_f)
            with open(npu_stack_path, 'w', encoding='utf8') as n_s_f:
                json.dump({}, n_s_f)
            with open(user_mapping_path, 'w', encoding='utf8') as u_m_f:
                yaml.safe_dump(user_mapping, u_m_f)

            stack_mode = True
            auto_analyze = True
            fuzzy_match = False
            dump_mode = Const.SUMMARY
            mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)
            mapping_config = MappingConfig(api_mapping=user_mapping_path)

            ms_comparator = MSComparator(mode_config, mapping_config)
            result_df = ms_comparator.compare_process((npu_dump_path, bench_dump_path, npu_stack_path))
            
            user_mapping_dict = {}
            for i in user_mapping:
                user_mapping_dict[i.get('ms_api')] = {'input': i.get('ms_args'), 'output': i.get('ms_output')}
            match_set = set()
            for key in npu_data.get('data').keys():
                matched_dict = user_mapping_dict.get(key.rsplit('.', 2)[0])
                match_set.update({key + '.input.' + str(i) for i in matched_dict.get('input')})
                match_set.update({key + '.output.' + str(i) for i in matched_dict.get('output')})

            self.assertTrue((result_df.loc[result_df['NPU Name'].isin(match_set), 'Bench Name'] != 'N/A').all())
            self.assertTrue((result_df.loc[~result_df['NPU Name'].isin(match_set), 'Bench Name'] == 'N/A').all())
        finally:
            shutil.rmtree(data_path)

    def test_load_internal_api(self):
        stack_mode = True
        auto_analyze = True
        fuzzy_match = False
        dump_mode = Const.SUMMARY

        mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)
        mapping_config = MappingConfig()

        ms_comparator = MSComparator(mode_config, mapping_config)
        api_dict = ms_comparator.load_internal_api()
        self.assertEqual(api_dict['Functional.abs'], 'Torch.abs')

    def test_process_cell_mapping(self):
        self.base_test_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.input_dir = os.path.join(self.base_test_dir, 'resources')
        cell_mapping_path = os.path.join(self.input_dir, 'common', 'cell_mapping.yaml')

        stack_mode = True
        auto_analyze = True
        fuzzy_match = False
        dump_mode = Const.SUMMARY

        mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)
        mapping_config = MappingConfig(cell_mapping=cell_mapping_path)

        ms_comparator = MSComparator(mode_config, mapping_config)
        npu_op_name = ms_comparator.process_cell_mapping(npu_cell_dict.get('op_name')[0])
        self.assertEqual(npu_op_name, 'Module.fc1.Linear.forward.0.input.0')

    def test_read_npy_data(self):
        stack_mode = True
        auto_analyze = True
        fuzzy_match = False
        dump_mode = Const.ALL

        mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)
        mapping_config = MappingConfig()

        ms_comparator = MSComparator(mode_config, mapping_config)

        self.temp_file = tempfile.NamedTemporaryFile(suffix='.pt')
        tensor = torch.Tensor([1, 2, 3])
        filename = self.temp_file.name.split('/')[-1]
        torch.save(tensor, self.temp_file.name)
        result = ms_comparator.read_npy_data('/tmp', filename, load_pt_file=True)
        self.assertTrue(np.array_equal(result, np.array([1, 2, 3])))
        self.temp_file.close()

        self.temp_file = tempfile.NamedTemporaryFile(suffix='.npy')
        tensor = np.array([1, 2, 3])
        filename = self.temp_file.name.split('/')[-1]
        np.save(self.temp_file.name, tensor)
        result = ms_comparator.read_npy_data('/tmp', filename, load_pt_file=False)
        self.assertTrue(np.array_equal(result, np.array([1, 2, 3])))
        self.temp_file.close()

    def test_process_internal_api_mapping(self):
        stack_mode = True
        auto_analyze = True
        fuzzy_match = False
        dump_mode = Const.ALL

        mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)
        mapping_config = MappingConfig(api_mapping=1)

        ms_comparator = MSComparator(mode_config, mapping_config)

        npu_op_name = "Mint.addcmul.0.forward.input.0"
        result = ms_comparator.process_internal_api_mapping(npu_op_name)
        self.assertEqual(result, "Torch.addcmul.0.forward.input.0")

        npu_op_name = "MintFunctional.addcmul.0.forward.input.0"
        result = ms_comparator.process_internal_api_mapping(npu_op_name)
        self.assertEqual(result, "Functional.addcmul.0.forward.input.0")

        npu_op_name = "Functional.abs"
        result = ms_comparator.process_internal_api_mapping(npu_op_name)
        self.assertEqual(result, "Torch.abs")

    def test_get_api_name(self):
        stack_mode = True
        auto_analyze = True
        fuzzy_match = False
        dump_mode = Const.ALL

        mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)
        mapping_config = MappingConfig()

        ms_comparator = MSComparator(mode_config, mapping_config)

        api_list = ["Functional", "absolute", "0", "forward", "input", "0"]
        result = ms_comparator.get_api_name(api_list)
        self.assertEqual(result, "Functional.absolute")

        api_list = ["Mint"]
        with self.assertRaises(CompareException):
            ms_comparator.get_api_name(api_list)