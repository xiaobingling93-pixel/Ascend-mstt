# coding=utf-8
import os
import torch
import unittest
import shutil
import numpy as np
from msprobe.pytorch.compare.pt_compare import PTComparator
from msprobe.core.common.utils import CompareException
from msprobe.test.core_ut.compare.test_acc_compare import npu_dict, bench_dict


base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_pt_compare')


def generate_npy(base_dir):
    data_path = os.path.join(base_dir, '1.npy')
    data = np.array([1, 2, 3, 4])
    np.save(data_path, data)


def generate_pt(base_dir):
    data_path = os.path.join(base_dir, '1.pt')
    data = torch.Tensor([1, 2, 3, 4])
    torch.save(data, data_path)


class TestUtilsMethods(unittest.TestCase):
    def setUp(self):
        os.makedirs(base_dir, mode=0o750, exist_ok=True)

    def tearDown(self):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

    def test_check_op(self):
        fuzzy_match = False
        pt_comparator = PTComparator()
        result = pt_comparator.check_op(npu_dict, bench_dict, fuzzy_match)
        self.assertEqual(result, True)

    def test_match_op(self):
        fuzzy_match = False
        pt_comparator = PTComparator()
        a, b = pt_comparator.match_op([npu_dict], [bench_dict], fuzzy_match)
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)

    def test_read_npy_data(self):
        generate_npy(base_dir)
        with self.assertRaises(CompareException) as context:
            result = PTComparator().read_npy_data(base_dir, '1.npy')
        self.assertEqual(context.exception.code, CompareException(CompareException.INVALID_FILE_ERROR))
