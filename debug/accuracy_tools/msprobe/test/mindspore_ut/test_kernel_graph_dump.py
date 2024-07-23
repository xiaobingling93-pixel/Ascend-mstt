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
from msprobe.mindspore.dump.kernel_graph_dump import KernelGraphDump


class TestKernelGraphDump(TestCase):

    def test_handle(self):
        json_config = {
            "task": "tensor",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [0, 2],
            "level": "L2"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        task_config.data_mode = ["output"]
        task_config.file_format = "bin"
        config = DebuggerConfig(common_config, task_config)
        dumper = KernelGraphDump(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["iteration"], "0|2")
        self.assertEqual(dumper.dump_json["common_dump_settings"]["file_format"], "bin")
        self.assertEqual(dumper.dump_json["common_dump_settings"]["input_output"], 2)

        with patch("msprobe.mindspore.dump.kernel_graph_dump.make_dump_path_if_not_exists"), \
             patch("msprobe.mindspore.dump.kernel_graph_dump.FileOpen"), \
             patch("msprobe.mindspore.dump.kernel_graph_dump.json.dump"), \
             patch("msprobe.mindspore.dump.kernel_graph_dump.logger.info"):

            os.environ["GRAPH_OP_RUN"] = "1"
            with self.assertRaises(Exception) as context:
                dumper.handle()
            self.assertEqual(str(context.exception), "Must run in graph mode, not kbk mode")
            if "GRAPH_OP_RUN" in os.environ:
                del os.environ["GRAPH_OP_RUN"]

            dumper.handle()
            self.assertIn("kernel_graph_dump.json", os.environ.get("MS_ACL_DUMP_CFG_PATH"))

        if "MINDSPORE_DUMP_CONFIG" in os.environ:
            del os.environ["MINDSPORE_DUMP_CONFIG"]
        if "MS_ACL_DUMP_CFG_PATH" in os.environ:
            del os.environ["MS_ACL_DUMP_CFG_PATH"]
