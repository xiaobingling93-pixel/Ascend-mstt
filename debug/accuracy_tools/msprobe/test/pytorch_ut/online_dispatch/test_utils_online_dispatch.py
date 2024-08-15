# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from msprobe.pytorch.online_dispatch.utils import COLOR_RED, COLOR_CYAN, COLOR_YELLOW, COLOR_RESET, COMPARE_LOGO, get_callstack, np_save_data, data_to_cpu, logger_debug, logger_info, logger_warn, logger_error, logger_user, logger_logo, DispatchException

cpu_device = torch._C.device("cpu")

class FakeData:
    def init(self):
        self.numpy=np.random.rand(5,5)

class FakeDataNoNumpy:
    def init(self):
        self.data=np.random.rand(5,5)

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.stack=inspect.stack()
        self.data_path=""
        self.file_name="data"
        self.data=FakeData()
        self.data_nonumpy=FakeDataNoNumpy()
        self.dispatch_exception=DispatchException(err_code=1, err_msg="messages")
        Path(os.path.join(self.data_path, f'{self.file_name}.npy')).touch()
    def tearDown(self):
        if os.path.exists(os.path.join(self.data_path, f'{self.file_name}.npy')):
            os.remove(os.path.join(self.data_path, f'{self.file_name}.npy'))

    @patch('msprobe.core.common.file_check.change_mode')
    def test_np_save_data_should_error_when_input_wrong(self,mock_change_mode):
        np_save_data(self.data_nonumpy,self.file_name,self.data_path)
        mock_change_mode.assert_not_called()

    def test_data_to_cpu_should_return_tensor_copy_when_input_tensor(self):
        data = torch.tensor([1,2],device=cpu_device,dtype=torch.float16)
        deep=1
        data_cpu=[]
        self.assertEqual(data_to_cpu(data,deep,data_cpu).all(),data.clone().detach().float().all())

    def test_data_to_cpu_should_return_list_when_input_list(self):
        data=[1,2]
        deep=0
        data_cpu=[]
        self.assertEqual(data_to_cpu(data,deep,data_cpu), data)

    @patch('msprobe.pytorch.online_dispatch.utils.get_mp_logger')
    def test_logger_debug(self,mock_inf0):
        logger_debug("messages")
        mock_inf0.return_value.assert_called_once_with("DEBUG messages")

    @patch('msprobe.pytorch.online_dispatch.utils.get_mp_logger')
    def test_logger_info(self,mock_info):
        logger_info("messages")
        mock_info.return_value.assert_called_once_with("INFO messages")

    @patch('msprobe.pytorch.online_dispatch.utils.get_mp_logger')
    def test_logger_warn(self,mock_info):
        logger_warn("messages")
        mock_info.return_value.assert_called_once_with(f'{COLOR_YELLOW}WARNING messages {COLOR_RESET}')

    @patch('msprobe.pytorch.online_dispatch.utils.get_mp_logger')
    def test_logger_error(self,mock_info):
        logger_error("messages")
        mock_info.return_value.assert_called_once_with(f'{COLOR_RED}ERROR messages {COLOR_RESET}')

    @patch('msprobe.pytorch.online_dispatch.utils.get_mp_logger')
    def test_logger_user(self,mock_info):
        logger_user("messages")
        mock_info.return_value.assert_called_once_with("messages")

    @patch('msprobe.pytorch.online_dispatch.utils.get_mp_logger')
    def test_logger_logo(self,mock_info):
        logger_logo()
        mock_info.return_value.assert_called_once_with(f'{COLOR_CYAN}{COMPARE_LOGO} {COLOR_RESET}')

    def test_str(self):
        self.assertEqual(self.dispatch_exception.__str__(),"messages")