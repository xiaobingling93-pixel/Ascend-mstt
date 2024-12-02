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

from msprobe.core.overflow_check.level import OverflowLevel


class TestOverflowLevel(unittest.TestCase):

    def test_enum_values(self):
        self.assertEqual(OverflowLevel.MEDIUM.value, "medium")
        self.assertEqual(OverflowLevel.HIGH.value, "high")
        self.assertEqual(OverflowLevel.CRITICAL.value, "critical")

    def test_enum_names(self):
        self.assertEqual(OverflowLevel.MEDIUM.name, "MEDIUM")
        self.assertEqual(OverflowLevel.HIGH.name, "HIGH")
        self.assertEqual(OverflowLevel.CRITICAL.name, "CRITICAL")

    def test_enum_iteration(self):
        levels = [level for level in OverflowLevel]
        self.assertEqual(levels, [OverflowLevel.MEDIUM, OverflowLevel.HIGH, OverflowLevel.CRITICAL])


if __name__ == "__main__":
    unittest.main()
