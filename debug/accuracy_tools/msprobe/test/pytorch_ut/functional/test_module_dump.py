# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.log import logger
from msprobe.pytorch import PrecisionDebugger
from msprobe.pytorch.service import torch_version_above_or_equal_2
from msprobe.pytorch.functional.module_dump import module_dump, module_dump_end, \
    hook_handle_list, remove_hook, register_hook
from msprobe.pytorch.hook_module.api_registry import api_register


class TestModuleDump(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api_register.api_originality()

    @classmethod
    def tearDownClass(cls):
        api_register.api_originality()

    def setUp(self):
        self.module = nn.Linear(8, 4)

    def tearDown(self):
        hook_handle_list.clear()

    @patch.object(logger, 'error')
    def test_module_dump(self, mock_error):
        with self.assertRaises(MsprobeException) as context:
            module_dump(1, "TestModule")
            self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
            mock_error.assert_called_with("The parameter module in module_dump must be a Module subclass.")

        with self.assertRaises(MsprobeException) as context:
            module_dump(self.module, 1)
            self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
            mock_error.assert_called_with("The parameter dump_name in module_dump must be a str type.")

        with patch('msprobe.pytorch.functional.module_dump.register_hook') as mock_register_hook:
            module_dump(self.module, "TestModule")
            mock_register_hook.assert_called_with(self.module, "TestModule")

    def test_module_dump_end(self):
        hook_handle_list.extend([1, 2, 3])
        with patch('msprobe.pytorch.functional.module_dump.remove_hook') as mock_remove_hook:
            module_dump_end()
            mock_remove_hook.assert_called_once()
        self.assertEqual(hook_handle_list, [])

    def test_register_hook(self):
        PrecisionDebugger(dump_path="./")
        register_hook(self.module, "TestModule")
        if torch_version_above_or_equal_2:
            self.assertEqual(len(hook_handle_list), 6)
        else:
            self.assertEqual(len(hook_handle_list), 5)

    def test_remove_hook(self):
        mock_handle = MagicMock(spec=torch.utils.hooks.RemovableHandle)
        hook_handle_list.append(mock_handle)
        remove_hook()

        mock_handle.remove.assert_called_once()
