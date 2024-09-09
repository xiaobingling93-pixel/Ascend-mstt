#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
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
"""
import unittest
from unittest.mock import patch, MagicMock

from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.debugger.precision_debugger import PrecisionDebugger
from msprobe.mindspore.runtime import Runtime
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.core.common.const import Const


class TestPrecisionDebugger(unittest.TestCase):

    @patch.object(DebuggerConfig, "_make_dump_path_if_not_exists")
    def test_start(self, _):
        class Handler:
            called = False

            def handle(self):
                Handler.called = True

        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L1"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        handler = Handler()

        mock_get_mode = MagicMock()
        mock_parse_json_config = MagicMock()
        with patch("msprobe.mindspore.debugger.precision_debugger.parse_json_config", new=mock_parse_json_config), \
             patch.object(PrecisionDebugger, "_get_execution_mode", new=mock_get_mode), \
             patch("msprobe.mindspore.debugger.precision_debugger.TaskHandlerFactory.create", return_value=handler):
            mock_get_mode.return_value = MsConst.GRAPH_GE_MODE
            mock_parse_json_config.return_value = [common_config, task_config]
            debugger = PrecisionDebugger()
            self.assertEqual(Runtime.step_count, 0)
            self.assertFalse(Runtime.is_running)
            debugger.start()
            self.assertTrue(Runtime.is_running)
            self.assertTrue(isinstance(debugger.config, DebuggerConfig))
            self.assertTrue(Handler.called)

            mock_get_mode.return_value = MsConst.PYNATIVE_MODE
            with patch("msprobe.mindspore.debugger.precision_debugger.Service") as mock_Service:
                debugger = PrecisionDebugger()
                debugger.start()
                service = mock_Service.return_value
                mock_Service.assert_called_with(debugger.config)
                service.start.assert_called_with(None)

        PrecisionDebugger._instance = None
        with self.assertRaises(Exception) as context:
            debugger.start()
        self.assertEqual(str(context.exception), "No instance of PrecisionDebugger found.")

        with patch("msprobe.mindspore.debugger.precision_debugger.parse_json_config", new=mock_parse_json_config), \
             patch.object(PrecisionDebugger, "_get_execution_mode", new=mock_get_mode), \
             patch("msprobe.mindspore.debugger.precision_debugger.TaskHandlerFactory.create", return_value=handler):
            common_config.task = Const.FREE_BENCHMARK
            mock_get_mode.return_value = MsConst.PYNATIVE_MODE
            mock_parse_json_config.return_value = [common_config, task_config]
            Handler.called = False
            debugger = PrecisionDebugger()
            debugger.start()
            self.assertTrue(Handler.called)

    def test_stop_step(self):
        class MockPrecisionDebugger:
            def __init__(self):
                self.task = Const.TENSOR
                self.service = None
        PrecisionDebugger._instance = None
        with self.assertRaises(Exception) as context:
            PrecisionDebugger.stop()
        self.assertEqual(str(context.exception), "PrecisionDebugger instance is not created.")
        with self.assertRaises(Exception) as context:
            PrecisionDebugger.step()
        self.assertEqual(str(context.exception), "PrecisionDebugger instance is not created.")
        PrecisionDebugger._instance = MockPrecisionDebugger()
        Runtime.is_running = True
        PrecisionDebugger.stop()
        self.assertFalse(Runtime.is_running)
        Runtime.step_count = 0
        PrecisionDebugger.step()
        self.assertEqual(Runtime.step_count, 1)
