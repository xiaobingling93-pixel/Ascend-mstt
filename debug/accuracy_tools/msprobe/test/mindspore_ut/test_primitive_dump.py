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
from unittest.mock import Mock, patch

from mindspore import nn

from msprobe.mindspore.service import Service
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig


class DummyModel(nn.Cell):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.dense = nn.Dense(2, 2)

    def construct(self, x):
        return self.dense(x)


class TestService(unittest.TestCase):
    @patch.object(DebuggerConfig, "_make_dump_path_if_not_exists")
    def setUp(self, _):
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
        self.service = Service(config)
        self.service.model = Mock()
        self.service.data_collector = Mock()
        self.service.switch = True  # Make sure the switch is on for testing

    def test_check_model_valid_none(self):
        model = None
        self.assertIsNone(self.service.check_model_valid(model))

    def test_check_model_valid_valid_model(self):
        model = DummyModel()
        self.assertEqual(self.service.check_model_valid(model), model)

    def test_check_model_valid_invalid_model(self):
        model = "invalid_model"
        with self.assertRaises(MsprobeException) as context:
            self.service.check_model_valid(model)

        # For the purpose of the test, let's also verify the expected exception message
        expected_message = "[msprobe] 无效参数： model 参数必须是 mindspore.nn.Cell 类型。"
        self.assertEqual(str(context.exception), expected_message)

    def test_update_primitive_counters(self):
        primitive_name = "test_primitive"
        self.service.update_primitive_counters(primitive_name)
        self.assertEqual(self.service.primitive_counters[primitive_name], 0)
        self.service.update_primitive_counters(primitive_name)
        self.assertEqual(self.service.primitive_counters[primitive_name], 1)
