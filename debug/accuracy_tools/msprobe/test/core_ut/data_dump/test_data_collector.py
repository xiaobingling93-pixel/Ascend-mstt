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
from unittest.mock import patch, mock_open, MagicMock

from msprobe.core.common.utils import Const
from msprobe.core.data_dump.data_collector import DataCollector
from msprobe.pytorch.debugger.debugger_config import DebuggerConfig
from msprobe.pytorch.pt_config import parse_json_config
from msprobe.core.data_dump.json_writer import DataWriter
from msprobe.core.data_dump.data_processor.base import BaseDataProcessor
from msprobe.core.data_dump.data_processor.pytorch_processor import StatisticsDataProcessor


class TestDataCollector(unittest.TestCase):
    def setUp(self):
        mock_json_data = {
            "dump_path": "./ut_dump",
        }
        with patch("msprobe.pytorch.pt_config.FileOpen", mock_open(read_data='')), \
                patch("msprobe.pytorch.pt_config.json.load", return_value=mock_json_data):
            common_config, task_config = parse_json_config("./config.json", Const.STATISTICS)
        config = DebuggerConfig(common_config, task_config, Const.STATISTICS, "./ut_dump", "L1")
        self.data_collector = DataCollector(config)

    def test_update_data(self):
        self.data_collector.config.task = Const.OVERFLOW_CHECK
        self.data_collector.data_processor.has_overflow = True
        with patch("msprobe.core.data_dump.json_writer.DataWriter.update_data", return_value=None):
            result1 = self.data_collector.update_data("test message", "test1:")
        self.assertEqual(result1, "test1:Overflow detected.")

        self.data_collector.data_processor.has_overflow = False
        result2 = self.data_collector.update_data("test message", "test2:")
        self.assertEqual(result2, "test2:No Overflow, OK.")

        self.data_collector.config.task = Const.STATISTICS
        self.data_collector.data_processor.has_overflow = True
        with patch("msprobe.core.data_dump.json_writer.DataWriter.update_data", return_value=None):
            result3 = self.data_collector.update_data("test message", "test3")
        self.assertEqual(result3, "test3")

    def test_pre_forward_data_collect(self):
        self.data_collector.check_scope_and_pid = MagicMock(return_value=False)
        self.data_collector.is_inplace = MagicMock(return_value=False)
        self.data_collector.data_processor.analyze_pre_forward = MagicMock()
        name = "TestModule.forward"
        pid = 123

        self.data_collector.pre_forward_data_collect(name, None, pid, None)
        self.data_collector.check_scope_and_pid.assert_called_once_with(
            self.data_collector.scope, "TestModule.backward", 123)

    def test_handle_data(self):
        with patch.object(DataCollector, "update_data", return_value="msg") as mock_update_data, \
             patch.object(DataCollector, "write_json") as mock_write_json, \
             patch("msprobe.core.data_dump.data_collector.logger.info") as mock_info, \
             patch("msprobe.core.data_dump.json_writer.DataWriter.flush_data_when_buffer_is_full") as mock_flush:
            self.data_collector.handle_data("Tensor.add", {"min": 0})
            msg = "msprobe is collecting data on Tensor.add. "
            mock_update_data.assert_called_with({"min": 0}, msg)
            mock_info.assert_called_with("msg", end='\r')
            mock_flush.assert_called()
            mock_write_json.assert_not_called()

            mock_update_data.reset_mock()
            mock_info.reset_mock()
            mock_flush.reset_mock()
            self.data_collector.handle_data("Tensor.add", {}, use_buffer=False)
            mock_update_data.assert_not_called()
            mock_info.assert_not_called()
            mock_write_json.assert_called()

    @patch.object(DataCollector, "update_construct")
    @patch.object(DataWriter, "update_stack")
    @patch.object(BaseDataProcessor, "analyze_api_call_stack")
    @patch.object(DataCollector, "handle_data")
    def test_forward_data_collect(self, mock_handle_data, _, __, ___):
        with patch.object(DataCollector, "check_scope_and_pid", return_value=True), \
             patch.object(DataCollector, "is_inplace", return_value=False), \
             patch.object(StatisticsDataProcessor, "analyze_forward", return_value={}):
            with patch.object(StatisticsDataProcessor, "is_terminated", return_value=True), \
                 self.assertRaises(Exception) as context:
                self.data_collector.forward_data_collect("name", "module", "pid", "module_input_output")
            mock_handle_data.assert_called_with("name", {}, use_buffer=False)
            self.assertEqual(str(context.exception), "[msprobe] exit")

            self.data_collector.forward_data_collect("name", "module", "pid", "module_input_output")
            mock_handle_data.assert_called_with("name", {})

    @patch.object(DataCollector, "update_construct")
    @patch.object(DataCollector, "handle_data")
    def test_backward_data_collect(self, mock_handle_data, _):
        with patch.object(DataCollector, "check_scope_and_pid", return_value=True), \
             patch.object(StatisticsDataProcessor, "analyze_backward", return_value={}):
            with patch.object(StatisticsDataProcessor, "is_terminated", return_value=True), \
                 self.assertRaises(Exception) as context:
                self.data_collector.backward_data_collect("name", "module", "pid", "module_input_output")
            mock_handle_data.assert_called_with("name", {}, use_buffer=False)
            self.assertEqual(str(context.exception), "[msprobe] exit")

            self.data_collector.backward_data_collect("name", "module", "pid", "module_input_output")
            mock_handle_data.assert_called_with("name", {})
