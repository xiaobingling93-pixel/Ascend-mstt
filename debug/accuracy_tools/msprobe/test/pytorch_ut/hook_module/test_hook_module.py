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
from unittest.mock import MagicMock, patch
import threading

from msprobe.pytorch.hook_module.hook_module import HOOKModule


class TestHOOKModuleInit(unittest.TestCase):

    def setUp(self):
        self.mock_build_hook = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))

    def test_thread_handling(self):
        module = HOOKModule(self.mock_build_hook)
        current_thread_id = module.current_thread
        self.assertEqual(current_thread_id, threading.current_thread().ident)


class TestHOOKModuleCall(unittest.TestCase):
    def setUp(self):
        self.mock_build_hook = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
        self.module = HOOKModule(self.mock_build_hook)

    @patch.object(HOOKModule, '_call_func')
    def test_call_function(self, mock_call_func):
        mock_call_func.return_value = "test_result"
        result = self.module("input_data")
        mock_call_func.assert_called_once_with("input_data", **{})
        self.assertEqual(result, "test_result")

    @patch.object(HOOKModule, '_call_func')
    def test_call_func_with_hooks(self, mock_call_func):
        mock_call_func.return_value = "test_result_with_hooks"
        result = self.module("input_data")
        self.assertEqual(result, "test_result_with_hooks")
        HOOKModule.inner_stop_hook[self.module.current_thread] = False
