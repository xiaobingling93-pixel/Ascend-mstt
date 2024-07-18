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

from atat.core.common_config import CommonConfig, BaseConfig
from atat.mindspore.debugger.debugger_config import DebuggerConfig
from atat.mindspore.overflow_check.overflow_check_tool_factory import OverflowCheckToolFactory


class TestOverflowCheckToolFactory(TestCase):

    def test_create(self):
        json_config = {
            "task": "overflow_check",
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L2"
        }

        common_config = CommonConfig(json_config)
        task_config = BaseConfig(json_config)
        config = DebuggerConfig(common_config, task_config)

        config.level = "module"
        with self.assertRaises(Exception) as context:
            OverflowCheckToolFactory.create(config)
        self.assertEqual(str(context.exception), "valid level is needed.")

        config.level = "cell"
        with self.assertRaises(Exception) as context:
            OverflowCheckToolFactory.create(config)
        self.assertEqual(str(context.exception), "Overflow check in not supported in this mode.")

        config.level = "kernel"
        dumper = OverflowCheckToolFactory.create(config)
        self.assertEqual(dumper.dump_json["common_dump_settings"]["file_format"], "npy")
