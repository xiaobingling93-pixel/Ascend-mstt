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

from msprobe.mindspore.free_benchmark.common.config import Config


class TestConfig(unittest.TestCase):
    def test_config(self):
        self.assertTrue(hasattr(Config, "is_enable"))
        self.assertTrue(hasattr(Config, "handler_type"))
        self.assertTrue(hasattr(Config, "pert_type"))
        self.assertTrue(hasattr(Config, "stage"))
        self.assertTrue(hasattr(Config, "dump_level"))
        self.assertTrue(hasattr(Config, "steps"))
        self.assertTrue(hasattr(Config, "ranks"))
        self.assertTrue(hasattr(Config, "dump_path"))
