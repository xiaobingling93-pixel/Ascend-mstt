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
import os

from unittest import TestCase
from unittest.mock import patch

from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.overflow_check.kernel_graph_overflow_check import KernelGraphOverflowCheck


class TestKernelGraphOverflowCheck(TestCase):
    @patch("msprobe.mindspore.debugger.debugger_config.create_directory")
    def test_handle(self, _):
        json_config = {
            "task": "overflow_check",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L2"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        task_config.check_mode = "atomic"
        config = DebuggerConfig(common_config, task_config)
        checker = KernelGraphOverflowCheck(config)
        self.assertEqual(checker.dump_json["common_dump_settings"]["op_debug_mode"], 2)

        os.environ["MS_ACL_DUMP_CFG_PATH"] = "path"
        with patch("msprobe.mindspore.overflow_check.kernel_graph_overflow_check.create_directory"), \
             patch("msprobe.mindspore.overflow_check.kernel_graph_overflow_check.FileOpen"), \
             patch("msprobe.mindspore.overflow_check.kernel_graph_overflow_check.json.dump"), \
             patch("msprobe.mindspore.overflow_check.kernel_graph_overflow_check.logger.info"):

            os.environ["GRAPH_OP_RUN"] = "1"
            with self.assertRaises(Exception) as context:
                checker.handle()
            self.assertEqual(str(context.exception), "Must run in graph mode, not kbk mode")
            if "GRAPH_OP_RUN" in os.environ:
                del os.environ["GRAPH_OP_RUN"]

            checker.handle()
            self.assertIn("kernel_graph_overflow_check.json", os.environ.get("MINDSPORE_DUMP_CONFIG"))
            self.assertEqual(os.environ.get("MS_ACL_DUMP_CFG_PATH"), None)

        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]
