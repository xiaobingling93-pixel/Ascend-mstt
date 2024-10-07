# coding=utf-8
import unittest
from msprobe.mindspore.compare.ms_compare import MSComparator
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
        dump_mode = Const.SUMMARY
        stack_json_data = {}
        ms_comparator = MSComparator(data_mapping=data_mapping)

        npu_ops_all = ms_comparator.merge_data(npu_json_data, stack_json_data, dump_mode)
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

        bench_ops_all = ms_comparator.merge_data(bench_json_data, stack_json_data, dump_mode)
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

        result = ms_comparator.get_accuracy(npu_ops_all, bench_ops_all, dump_mode)
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
                                                         False, dump_mode)
        self.assertListEqual(result_df.values.tolist(), [])
