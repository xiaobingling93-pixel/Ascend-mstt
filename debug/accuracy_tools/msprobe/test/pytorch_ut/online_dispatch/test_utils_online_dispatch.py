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

from msprobe.pytorch.online_dispatch.utils import COLOR_RED, COLOR_CYAN, COLOR_YELLOW, COLOR_RESET, COMPARE_LOGO, data_to_cpu, DispatchException

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

    def test_str(self):
        self.assertEqual(self.dispatch_exception.__str__(),"messages")