# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
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
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn

from msprobe.core.data_dump.api_registry import ApiRegistry
from msprobe.pytorch import PrecisionDebugger
from msprobe.pytorch.hook_module.api_register import get_api_register
from msprobe.pytorch.service import torch_version_above_or_equal_2


class TestModuleDumper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        PrecisionDebugger._instance = None
        get_api_register().restore_all_api()

    @classmethod
    def tearDownClass(cls):
        PrecisionDebugger._instance = None
        get_api_register().restore_all_api()

    def setUp(self):
        self.module = nn.Linear(8, 4)
        debugger = PrecisionDebugger(dump_path="./")
        self.module_dumper = debugger.module_dumper

    def test_stop_module_dump(self):
        self.module_dumper.hook_handle_list.extend([1, 2, 3])
        with patch.object(ApiRegistry, 'register_all_api') as mock_api_register:
            mock_handle1 = MagicMock(spec=torch.utils.hooks.RemovableHandle)
            mock_handle2 = MagicMock(spec=torch.utils.hooks.RemovableHandle)
            self.module_dumper.hook_handle_list.extend([mock_handle1, mock_handle2])

            self.module_dumper.stop_module_dump()
            mock_handle1.remove.assert_called_once()
            mock_handle2.remove.assert_called_once()
            self.assertEqual(self.module_dumper.hook_handle_list, [])
            mock_api_register.assert_called_once()

    def test_register_hook(self):
        self.module_dumper.register_hook(self.module, "TestModule")
        if torch_version_above_or_equal_2:
            self.assertEqual(len(self.module_dumper.hook_handle_list), 6)
        else:
            self.assertEqual(len(self.module_dumper.hook_handle_list), 5)
