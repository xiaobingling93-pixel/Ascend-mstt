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

from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.free_benchmark.common.config import Config


class TestConfig(unittest.TestCase):
    def test_config(self):
        self.assertFalse(Config.is_enable)
        self.assertEqual(Config.handler_type, FreeBenchmarkConst.DEFAULT_HANDLER_TYPE)
        self.assertEqual(Config.pert_type, FreeBenchmarkConst.DEFAULT_PERT_TYPE)
        self.assertEqual(Config.stage, FreeBenchmarkConst.DEFAULT_STAGE)
        self.assertEqual(Config.dump_level, FreeBenchmarkConst.DEFAULT_DUMP_LEVEL)
        self.assertEqual(Config.steps, [])
        self.assertEqual(Config.ranks, [])
        self.assertEqual(Config.dump_path, "")
