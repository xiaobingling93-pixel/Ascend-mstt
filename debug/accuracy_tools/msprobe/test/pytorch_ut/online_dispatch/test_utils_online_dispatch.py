# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


import unittest
import inspect
import numpy as np
import os
import sys
import torch
import psutil
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

from msprobe.pytorch.online_dispatch.utils import get_callstack, COLOR_RED, COLOR_CYAN, COLOR_YELLOW, COLOR_RESET, \
    COMPARE_LOGO, data_to_cpu, DispatchException, get_sys_info

cpu_device = torch._C.device("cpu")


class FakeData:
    def init(self):
        self.numpy = np.random.rand(5, 5)


class FakeDataNoNumpy:
    def init(self):
        self.data = np.random.rand(5, 5)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.stack = inspect.stack()
        self.data_path = ""
        self.file_name = "data"
        self.data = FakeData()
        self.data_nonumpy = FakeDataNoNumpy()
        self.dispatch_exception = DispatchException(err_code=1, err_msg="messages")
        Path(os.path.join(self.data_path, f'{self.file_name}.npy')).touch()

    def tearDown(self):
        if os.path.exists(os.path.join(self.data_path, f'{self.file_name}.npy')):
            os.remove(os.path.join(self.data_path, f'{self.file_name}.npy'))

    @patch('inspect.stack')
    def test_get_callstack(self, mock_stack):
        mock_stack.return_value = [
            (None, '/path/to/file1.py', 10, 'function_a', ['line_of_code_a'], None),
            (None, '/path/to/file2.py', 20, 'function_b', ['line_of_code_b'], None),
            (None, '/path/to/file3.py', 30, 'function_c', ['line_of_code_c'], None),
        ]

        result = get_callstack()

        expected_callstack = [
            ['/path/to/file3.py', '30', 'function_c', 'line_of_code_c']
        ]

        self.assertEqual(result, expected_callstack)

    def test_data_to_cpu_tensor(self):
        data = torch.tensor([1, 2], device=cpu_device, dtype=torch.float16)
        deep = 1
        data_cpu = []
        self.assertEqual(data_to_cpu(data, deep, data_cpu).all(), data.clone().detach().float().all())

    def test_data_to_cpu_list(self):
        data = [1, 2]
        deep = 0
        data_cpu = []
        self.assertEqual(data_to_cpu(data, deep, data_cpu), data)

    def test_data_to_cpu_tuple(self):
        data = (1, 2)
        deep = 0
        data_cpu = []
        self.assertEqual(data_to_cpu(data, deep, data_cpu), data)

    def test_data_to_cpu_dict(self):
        data = {1: 2}
        deep = 0
        data_cpu = []
        self.assertEqual(data_to_cpu(data, deep, data_cpu), data)

    def test_data_to_cpu_c_device(self):
        data = torch.device("cpu")
        deep = 0
        data_cpu = []
        result = data_to_cpu(data, deep, data_cpu)
        self.assertEqual(result, cpu_device)

    def test_str(self):
        self.assertEqual(self.dispatch_exception.__str__(), "messages")

    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_get_sys_info(self, mock_cpu_percent, mock_virtual_memory):
        mock_virtual_memory.return_value = MagicMock(
            total=8 * 1024 * 1024 * 1024,
            available=4 * 1024 * 1024 * 1024,
            used=3 * 1024 * 1024 * 1024
        )
        mock_cpu_percent.return_value = 55.0

        result = get_sys_info()

        expected_output = 'Total: 8192.00MB Free: 4096.00 MB Used: 3072.00 MB CPU: 55.0%'
        self.assertIn(expected_output, result)
