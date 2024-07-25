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
from unittest import TestCase
from unittest.mock import patch

from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.debugger.precision_debugger import PrecisionDebugger


class TestPrecisionDebugger(TestCase):
    def test_start(self):
        class Handler:
            called = False

            def handle(self):
                Handler.called = True

        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L0"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        handler = Handler()

        with patch("msprobe.mindspore.debugger.precision_debugger.parse_json_config",
                   return_value=[common_config, task_config]), \
             patch("msprobe.mindspore.debugger.precision_debugger.TaskHandlerFactory.create", return_value=handler):
            debugger = PrecisionDebugger()
            debugger.start()
        self.assertTrue(isinstance(debugger.config, DebuggerConfig))
        self.assertTrue(Handler.called)

        PrecisionDebugger._instance = None
        with self.assertRaises(Exception) as context:
            debugger.start()
        self.assertEqual(str(context.exception), "No instance of PrecisionDebugger found.")
