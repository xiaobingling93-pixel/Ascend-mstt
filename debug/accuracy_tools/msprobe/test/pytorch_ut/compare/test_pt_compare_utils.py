import os
import shutil
import threading
import unittest

import numpy as np

from msprobe.pytorch.compare.utils import read_pt_data
from msprobe.test.core_ut.compare.test_acc_compare import generate_pt


base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_pt_compare_utils_data')
pt_dir = os.path.join(base_dir, f'dump_data_dir')


class TestReadPtData(unittest.TestCase):

    def setUp(self):
        os.makedirs(base_dir, mode=0o750, exist_ok=True)
        os.makedirs(pt_dir, mode=0o750, exist_ok=True)

        self.lock = threading.Lock()

    def tearDown(self):
        if os.path.exists(pt_dir):
            shutil.rmtree(pt_dir)
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

    def test_read_pt_data(self):
        generate_pt(pt_dir)
        result = read_pt_data(pt_dir, 'Functional.linear.0.forward.input.0.pt')
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(np.array_equal(result, expected))
