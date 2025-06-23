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

import threading
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

from msprobe.core.hook_manager import HookSet
from msprobe.pytorch.hook_module.hook_module import HOOKModule


class TestHOOKModule(unittest.TestCase):
    def setUp(self):
        self.mock_build_hook = MagicMock(return_value=HookSet(MagicMock(), MagicMock(), MagicMock()))
        HOOKModule.module_count = defaultdict(int)
        HOOKModule.inner_stop_hook = defaultdict(bool)

    def test_init_with_stop_hook(self):
        expected_thread = threading.get_ident()
        HOOKModule.inner_stop_hook[expected_thread] = True

        module1 = HOOKModule(self.mock_build_hook)
        current_thread = module1.tid

        self.assertEqual(current_thread, expected_thread)
        self.assertTrue(module1.inner_stop_hook[current_thread])
        self.assertTrue(module1.stop_hook)
        self.assertFalse(hasattr(module1, "forward_data_collected"))

    def test_init_with_start_hook(self):
        module1 = HOOKModule(self.mock_build_hook)
        current_thread = module1.tid
        expected_thread = threading.get_ident()

        self.assertEqual(current_thread, expected_thread)
        self.assertFalse(module1.inner_stop_hook[current_thread])
        self.assertFalse(module1.stop_hook)
        self.assertTrue(hasattr(module1, "forward_data_collected"))

    @patch.object(HOOKModule, '_call_func')
    def test_call_with_stop_hooks(self, mock_call_func):
        mock_call_func.return_value = "test_result"
        expected_thread = threading.current_thread().ident
        HOOKModule.inner_stop_hook[expected_thread] = True

        module1 = HOOKModule(self.mock_build_hook)
        self.assertTrue(module1.stop_hook)

        result = module1("arg1", "arg2", key="value")
        mock_call_func.assert_called_once_with("arg1", "arg2", key="value")
        self.assertEqual(result, "test_result")
        self.assertTrue(HOOKModule.inner_stop_hook[expected_thread])

    @patch.object(HOOKModule, '_call_func')
    def test_call_with_start_hooks(self, mock_call_func):
        mock_call_func.return_value = "test_result"
        expected_thread = threading.current_thread().ident

        module1 = HOOKModule(self.mock_build_hook)
        self.assertFalse(module1.stop_hook)

        result = module1("arg1", "arg2", key="value")
        mock_call_func.assert_called_once_with("arg1", "arg2", key="value")
        self.assertEqual(result, "test_result")
        self.assertFalse(HOOKModule.inner_stop_hook[expected_thread])

    def test_reset_module_stats(self):
        HOOKModule.module_count = {"Tensor.add.0.forward": 0}
        HOOKModule.reset_module_stats()
        self.assertDictEqual(HOOKModule.module_count, defaultdict(int))

    def test_add_module_count(self):
        HOOKModule.add_module_count("Tensor.add.0.forward")
        self.assertEqual(HOOKModule.module_count["Tensor.add.0.forward"], 1)

    def test_get_module_count(self):
        HOOKModule.module_count = {"Tensor.add.0.forward": 0}
        result = HOOKModule.get_module_count("Tensor.add.0.forward")
        self.assertEqual(result, 0)
