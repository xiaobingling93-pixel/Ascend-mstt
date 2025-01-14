# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
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
from unittest.mock import patch, MagicMock

from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.core.common.const import Const, MsgConst
from msprobe.mindspore.cell_processor import CellProcessor
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.debugger.precision_debugger import PrecisionDebugger
from msprobe.mindspore.dump.hook_cell.hook_cell import HOOKCell
from msprobe.mindspore.runtime import Runtime


class TestPrecisionDebugger(unittest.TestCase):

    @patch("msprobe.mindspore.debugger.debugger_config.create_directory")
    def test_start(self, _):
        PrecisionDebugger._instance = None

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
             patch("msprobe.mindspore.debugger.precision_debugger.TaskHandlerFactory.create", return_value=handler), \
             patch("msprobe.mindspore.debugger.precision_debugger.set_register_backward_hook_functions"):
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
            with patch("msprobe.mindspore.debugger.precision_debugger.Service") as mock_Service, \
                 patch("msprobe.mindspore.debugger.precision_debugger.set_register_backward_hook_functions"):
                debugger = PrecisionDebugger()
                debugger.start()
                service = mock_Service.return_value
                mock_Service.assert_called_with(debugger.config)
                service.start.assert_called_with(None)

        PrecisionDebugger._instance = None
        with self.assertRaises(Exception) as context:
            debugger.start()
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)

        with patch("msprobe.mindspore.debugger.precision_debugger.parse_json_config", new=mock_parse_json_config), \
             patch.object(PrecisionDebugger, "_get_execution_mode", new=mock_get_mode), \
             patch("msprobe.mindspore.debugger.precision_debugger.TaskHandlerFactory.create", return_value=handler), \
             patch("msprobe.mindspore.debugger.precision_debugger.set_register_backward_hook_functions"):
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
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)
        with self.assertRaises(Exception) as context:
            PrecisionDebugger.step()
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)
        PrecisionDebugger._instance = MockPrecisionDebugger()
        Runtime.is_running = True
        PrecisionDebugger.stop()
        self.assertFalse(Runtime.is_running)
        Runtime.step_count = 0
        PrecisionDebugger.step()
        self.assertEqual(Runtime.step_count, 1)
        Runtime.step_count = 0

        HOOKCell.cell_count["api"] = 1
        PrecisionDebugger.step()
        self.assertEqual(HOOKCell.cell_count["api"], 0)

        with patch.object(CellProcessor, "reset_cell_stats") as mock_reset_cell:
            PrecisionDebugger.step()
        mock_reset_cell.assert_called_once()

    def test_forward_backward_dump_end(self):
        with patch("msprobe.mindspore.debugger.precision_debugger.set_register_backward_hook_functions"):
            debugger = PrecisionDebugger()
        debugger.task = "statistics"
        debugger.service = MagicMock()
        debugger.forward_backward_dump_end()
        debugger.service.stop.assert_called_once()
