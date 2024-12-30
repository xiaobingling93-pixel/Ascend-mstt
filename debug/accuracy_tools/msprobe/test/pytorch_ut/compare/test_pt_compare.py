# coding=utf-8
import os
import shutil
import unittest

import numpy as np
import torch

from msprobe.core.common.const import Const
from msprobe.core.common.utils import CompareException
from msprobe.core.compare.acc_compare import ModeConfig
from msprobe.pytorch.compare.pt_compare import PTComparator, compare
from msprobe.test.core_ut.compare.test_acc_compare import generate_dump_json, generate_stack_json


base_dir1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_pt_compare1')
base_dir2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_pt_compare2')


def generate_bf16_pt(base_dir):
    data_path = os.path.join(base_dir, 'bf16.pt')
    data = torch.tensor([1, 2, 3, 4], dtype=torch.bfloat16)
    torch.save(data, data_path)


def generate_dict_pt(base_dir):
    data_path = os.path.join(base_dir, 'dict.pt')
    torch.save({1: 0}, data_path)


class TestUtilsMethods(unittest.TestCase):

    def setUp(self):
        os.makedirs(base_dir1, mode=0o750, exist_ok=True)
        os.makedirs(base_dir2, mode=0o750, exist_ok=True)

    def tearDown(self):
        if os.path.exists(base_dir1):
            shutil.rmtree(base_dir1)
        if os.path.exists(base_dir2):
            shutil.rmtree(base_dir2)

    def test_read_npy_data_bf16(self):
        generate_bf16_pt(base_dir1)

        stack_mode = True
        auto_analyze = True
        fuzzy_match = False
        dump_mode = Const.ALL
        mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)

        pt_comparator = PTComparator(mode_config)
        result = pt_comparator.read_npy_data(base_dir1, 'bf16.pt')

        target_result = torch.tensor([1, 2, 3, 4], dtype=torch.float32).numpy()
        self.assertTrue(np.array_equal(result, target_result))

    def test_read_npy_data_dict(self):
        generate_dict_pt(base_dir1)

        stack_mode = True
        auto_analyze = True
        fuzzy_match = False
        dump_mode = Const.ALL
        mode_config = ModeConfig(stack_mode, auto_analyze, fuzzy_match, dump_mode)

        pt_comparator = PTComparator(mode_config)

        with self.assertRaises(CompareException) as context:
            result = pt_comparator.read_npy_data(base_dir1, 'dict.pt')
        self.assertEqual(context.exception.code, CompareException.DETACH_ERROR)

    def test_compare(self):
        generate_dump_json(base_dir2)
        generate_stack_json(base_dir2)

        dump_path = os.path.join(base_dir2, 'dump.json')

        input_param = {
            'npu_json_path': dump_path,
            'bench_json_path': dump_path,
            'is_print_compare_log': True
        }
        output_path = base_dir2

        compare(input_param, output_path)
        output_files = os.listdir(output_path)
        self.assertTrue(any(f.endswith(".xlsx") for f in output_files))

        input_param2 = {
            'npu_json_path': '',
            'bench_json_path': dump_path,
            'is_print_compare_log': True
        }
        with self.assertRaises(CompareException) as context:
            compare(input_param2, output_path)
        self.assertEqual(context.exception.code, 1)
