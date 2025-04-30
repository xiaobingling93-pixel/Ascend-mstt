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
                patch("msprobe.pytorch.pt_config.load_json", return_value=mock_json_data):
            common_config, task_config = parse_json_config("./config.json", Const.STATISTICS)
        config = DebuggerConfig(common_config, task_config, Const.STATISTICS, "./ut_dump", "L1")
        self.data_collector = DataCollector(config)

    def test_update_data(self):
        self.data_collector.config.task = Const.OVERFLOW_CHECK
        self.data_collector.data_processor.has_overflow = True
        with patch("msprobe.core.data_dump.json_writer.DataWriter.update_data") as mock_update_data, \
                patch("msprobe.core.data_dump.data_collector.logger.warning") as mock_warning, \
                patch("msprobe.core.data_dump.data_collector.logger.debug") as mock_debug:
            self.data_collector.update_data("Tensor.add", {"mean": 0})
            mock_update_data.assert_called_once_with({"mean": 0})
            mock_warning.assert_called_once_with("msprobe is collecting data on Tensor.add. Overflow detected.")
            mock_debug.assert_not_called()

            mock_update_data.reset_mock()
            mock_warning.reset_mock()
            mock_debug.reset_mock()

            self.data_collector.config.task = Const.STATISTICS
            self.data_collector.update_data("Tensor.add", {"mean": 0})
            mock_update_data.assert_called_once_with({"mean": 0})
            mock_warning.assert_not_called()
            mock_debug.assert_called_once_with("msprobe is collecting data on Tensor.add.")

    def test_handle_data(self):
        with patch.object(DataCollector, "update_data") as mock_update_data, \
                patch.object(DataCollector, "write_json") as mock_write_json, \
                patch("msprobe.core.data_dump.json_writer.DataWriter.flush_data_periodically") as mock_flush:
            self.data_collector.handle_data("Tensor.add", {"min": 0})
            mock_update_data.assert_called_with("Tensor.add", {"min": 0})

            mock_flush.assert_called()
            mock_write_json.assert_not_called()

            mock_update_data.reset_mock()
            mock_flush.reset_mock()
            self.data_collector.handle_data("Tensor.add", {}, flush=True)
            mock_update_data.assert_not_called()
            mock_flush.assert_not_called()
            mock_write_json.assert_called()

    @patch.object(DataCollector, "update_construct")
    @patch.object(DataWriter, "update_stack")
    @patch.object(BaseDataProcessor, "analyze_api_call_stack")
    @patch.object(DataCollector, "handle_data")
    def test_forward_data_collect(self, mock_handle_data, _, __, ___):
        with patch.object(DataCollector, "check_scope_and_pid", return_value=True), \
                patch.object(StatisticsDataProcessor, "analyze_forward", return_value={}):
            with patch.object(StatisticsDataProcessor, "is_terminated", new=True):
                self.data_collector.forward_data_collect("name", "module", "pid", "module_input_output")
                mock_handle_data.assert_called_with("name", {}, flush=True)

            self.data_collector.forward_data_collect("name", "module", "pid", "module_input_output")
            mock_handle_data.assert_called_with("name", {}, flush=False)

    @patch.object(DataCollector, "update_construct")
    @patch.object(DataCollector, "handle_data")
    def test_backward_data_collect(self, mock_handle_data, _):
        with patch.object(DataCollector, "check_scope_and_pid", return_value=True), \
                patch.object(StatisticsDataProcessor, "analyze_backward", return_value={}):
            with patch.object(StatisticsDataProcessor, "is_terminated", new=True):
                self.data_collector.backward_data_collect("name", "module", "pid", "module_input_output")
                mock_handle_data.assert_called_with("name", {}, flush=True)

            self.data_collector.backward_data_collect("name", "module", "pid", "module_input_output")
            mock_handle_data.assert_called_with("name", {}, flush=False)

    @patch.object(DataWriter, "update_debug")
    @patch.object(BaseDataProcessor, "analyze_debug_forward", return_value="data_info")
    def test_debug_data_collect_forward(self, _, mock_update_debug):
        self.data_collector.debug_data_collect_forward("variable", "name_with_count")
        mock_update_debug.assert_called_with({"name_with_count": "data_info"})

    @patch.object(DataWriter, "update_debug")
    @patch.object(BaseDataProcessor, "analyze_debug_backward")
    @patch.object(BaseDataProcessor, "analyze_element_to_all_none", return_value = "all_none_data_info")
    def test_debug_data_collect_backward(self, _, mock_analyze_debug_backward, mock_update_debug):
        self.data_collector.data_writer.cache_debug = {"data": None}
        self.data_collector.debug_data_collect_backward("variable", "name_with_count")
        mock_update_debug.assert_called_with({"name_with_count.debug": "all_none_data_info"})
        mock_analyze_debug_backward.assert_called_with("variable", "name_with_count.debug", self.data_collector.data_writer.cache_debug['data'])
        self.data_collector.data_writer.cache_debug = None
