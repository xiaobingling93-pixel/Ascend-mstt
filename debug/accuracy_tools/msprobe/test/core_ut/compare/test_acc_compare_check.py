# coding=utf-8
import unittest
import os
import tempfile

from msprobe.core.compare.check import check_dump_json_str, check_json_key_value, valid_key_value, \
    check_stack_json_str, check_mix_compare
from msprobe.core.common.utils import CompareException


# test_check_struct_match
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


# test_check_type_shape_match
npu_struct = [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]), ('torch.float32', [16])]
bench_struct = [('torch.float32', [1, 1, 28, 28]), ('torch.float32', [16, 1, 5, 5]), ('torch.float32', [16])]


# test_check_dump_json_str
op_data = {'input_args': [
    {
        'type': '@torch.Tensor',
        'dtype': 'torch.float32',
        'shape': [2, 2],
        'Max': 0.5,
        'Min': -0.7,
        'Mean': 0.2,
        'Norm': 0.9,
        'requires_grad': True,
        'data_name': 'Function.linear.0.forward.output.0.pt'
    }
]}


# test_check_json_key_value
input_output = {
    'type': '@torch.Tensor',
    'dtype': 'torch.float32',
    'shape': [2, 2],
    'Max': 0.5,
    'Min': -0.7,
    'Mean': 0.2,
    'Norm': 0.9,
    'requires_grad': True,
    'data_name': 'Function.linear.0.forward.output.0.pt'}


op_name = 'Functional.conv2d.0.backward.input.0'


class TestUtilsMethods(unittest.TestCase):
    def test_check_dump_json_str(self):
        with self.assertRaises(CompareException) as context:
            check_dump_json_str(op_data, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_json_key_value(self):
        with self.assertRaises(CompareException) as context:
            check_json_key_value(input_output, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_json_key_value_max_depth(self):
        result = check_json_key_value(input_output, op_name, depth=11)
        self.assertEqual(result, None)

    def test_valid_key_value_type_shape(self):
        key = 'shape'
        value = 'abc'
        with self.assertRaises(CompareException) as context:
            valid_key_value(key, value, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_OBJECT_TYPE_ERROR)

    def test_valid_key_value_type_requires_grad(self):
        key = 'requires_grad'
        value = 'abc'
        with self.assertRaises(CompareException) as context:
            valid_key_value(key, value, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_OBJECT_TYPE_ERROR)

    def test_check_stack_json_str_type_stack_info(self):
        stack_info = 'File'
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_OBJECT_TYPE_ERROR)

    def test_check_stack_json_str_2(self):
        stack_info = ['=File', 'File']
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_stack_json_str_3(self):
        stack_info = ['+File', 'File']
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_stack_json_str_4(self):
        stack_info = ['-File', 'File']
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)

    def test_check_stack_json_str_5(self):
        stack_info = ['@File', 'File']
        with self.assertRaises(CompareException) as context:
            check_stack_json_str(stack_info, op_name)
        self.assertEqual(context.exception.code, CompareException.INVALID_CHAR_ERROR)


class TestCheckMixCompare(unittest.TestCase):
    @staticmethod
    def create_temp_dir_with_files(file_names):
        temp_dir = tempfile.TemporaryDirectory()
        for name in file_names:
            open(os.path.join(temp_dir.name, name), 'w').close()
        return temp_dir

    def test_both_dirs_match(self):
        # 两个路径都包含完整的 MIX_DUMP_NAMES
        files = ['graph', 'pynative']
        npu_dir = self.create_temp_dir_with_files(files)
        bench_dir = self.create_temp_dir_with_files(files)

        self.assertTrue(check_mix_compare(npu_dir.name, bench_dir.name))

        npu_dir.cleanup()
        bench_dir.cleanup()

    def test_one_dir_missing_file(self):
        # 一个路径缺少文件
        npu_dir = self.create_temp_dir_with_files(['graph', 'pynative'])
        bench_dir = self.create_temp_dir_with_files(['graph'])

        self.assertFalse(check_mix_compare(npu_dir.name, bench_dir.name))

        npu_dir.cleanup()
        bench_dir.cleanup()

    def test_extra_file(self):
        # 有额外的文件，也应该返回 False
        npu_dir = self.create_temp_dir_with_files(['graph', 'pynative', 'extra'])
        bench_dir = self.create_temp_dir_with_files(['graph', 'pynative'])

        self.assertFalse(check_mix_compare(npu_dir.name, bench_dir.name))

        npu_dir.cleanup()
        bench_dir.cleanup()

    def test_both_dirs_mismatch(self):
        # 两个路径都不匹配
        npu_dir = self.create_temp_dir_with_files(['a'])
        bench_dir = self.create_temp_dir_with_files(['b', 'c'])

        self.assertFalse(check_mix_compare(npu_dir.name, bench_dir.name))

        npu_dir.cleanup()
        bench_dir.cleanup()
