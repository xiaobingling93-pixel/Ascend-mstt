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

from msprobe.mindspore.common.const import Const
from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_tool_factory import DumpToolFactory


class TestDumpToolFactory(TestCase):
    @patch("msprobe.mindspore.debugger.debugger_config.create_directory")
    def test_create(self, _):
        json_config = {
            "task": "statistics",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L1"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)

        config.level = "module"
        with self.assertRaises(Exception) as context:
            DumpToolFactory.create(config)
        self.assertEqual(str(context.exception), "Valid level is needed.")

        config.level = Const.KERNEL
        with self.assertRaises(Exception) as context:
            DumpToolFactory.create(config)
        self.assertEqual(str(context.exception), "Data dump is not supported in None mode when dump level is kernel.")

        config.execution_mode = Const.GRAPH_GE_MODE
        config.level = Const.CELL
        with self.assertRaises(Exception) as context:
            DumpToolFactory.create(config)
        self.assertEqual(str(context.exception), "Data dump is not supported in graph_ge mode when dump level is cell.")

        config.execution_mode = Const.GRAPH_KBYK_MODE
        config.level = Const.KERNEL
        dumper = DumpToolFactory.create(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["net_name"], "Net")
