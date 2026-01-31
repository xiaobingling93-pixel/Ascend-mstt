#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


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
