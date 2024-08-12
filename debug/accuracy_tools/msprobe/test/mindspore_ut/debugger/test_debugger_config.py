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
from unittest.mock import patch

from msprobe.core.common.const import Const, FileCheckConst
from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig


class TestDebuggerConfig(unittest.TestCase):
    @patch.object(DebuggerConfig, "_make_dump_path_if_not_exists")
    def test_init(self, _):
        json_config = {
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L2"
        }
        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        debugger_config = DebuggerConfig(common_config, task_config)
        self.assertEqual(debugger_config.task, Const.STATISTICS)
        self.assertEqual(debugger_config.file_format, "npy")
        self.assertEqual(debugger_config.check_mode, "all")
        self.assertEqual(debugger_config.overflow_nums, 1)

        common_config.dump_path = "./path"
        with self.assertRaises(Exception) as context:
            DebuggerConfig(common_config, task_config)
        self.assertEqual(str(context.exception), "Dump path must be absolute path.")

        common_config.dump_path = "./path"
        with self.assertRaises(Exception) as context:
            DebuggerConfig(common_config, task_config)
        self.assertEqual(str(context.exception), "Dump path must be absolute path.")

        common_config.level = "L1"
        common_config.task = Const.FREE_BENCHMARK
        debugger_config = DebuggerConfig(common_config, task_config)
        self.assertEqual(debugger_config.pert_type, FreeBenchmarkConst.DEFAULT_PERT_TYPE)
        self.assertEqual(debugger_config.handler_type, FreeBenchmarkConst.DEFAULT_HANDLER_TYPE)
        self.assertEqual(debugger_config.dump_level, FreeBenchmarkConst.DEFAULT_DUMP_LEVEL)
        self.assertEqual(debugger_config.stage, FreeBenchmarkConst.DEFAULT_STAGE)

        task_config.handler_type = FreeBenchmarkConst.FIX
        task_config.pert_mode = FreeBenchmarkConst.ADD_NOISE
        with self.assertRaises(Exception) as context:
            DebuggerConfig(common_config, task_config)
        self.assertEqual(str(context.exception),
                         "pert_mode must be improve_precision or empty when handler_type is fix, "
                         f"but got {FreeBenchmarkConst.ADD_NOISE}.")

    @patch("msprobe.mindspore.debugger.debugger_config.os.path.exists", return_value=False)
    def test__make_dump_path_if_not_exists(self, _):
        json_config = {"dump_path": "/absolute_path"}
        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        with patch("msprobe.mindspore.debugger.debugger_config.check_path_before_create") as mock_check_path, \
                patch("msprobe.mindspore.debugger.debugger_config.Path.mkdir") as mock_mkdir, \
                patch("msprobe.mindspore.debugger.debugger_config.FileChecker") as mock_checker:
            DebuggerConfig(common_config, task_config)
        mock_check_path.assert_called_with(json_config.get("dump_path"))
        mock_mkdir.assert_called_with(mode=FileCheckConst.DATA_DIR_AUTHORITY, exist_ok=True)
        mock_checker.assert_called_with(common_config.dump_path, FileCheckConst.DIR)
