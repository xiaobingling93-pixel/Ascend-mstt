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

from msprobe.core.common.const import Const
from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig


class TestDebuggerConfig(unittest.TestCase):
    @patch("msprobe.mindspore.debugger.debugger_config.create_directory")
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
