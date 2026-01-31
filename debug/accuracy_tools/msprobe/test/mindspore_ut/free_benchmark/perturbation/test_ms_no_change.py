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

import mindspore as ms
from mindspore import Tensor, ops

from msprobe.mindspore.free_benchmark.perturbation.no_change import NoChangePerturbation
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.core.common.const import Const


class TestNoChangePerturbation(unittest.TestCase):

    no_change_pert = None

    @classmethod
    def setUpClass(cls):
        cls.no_change_pert = NoChangePerturbation("mindspore.ops.add")

    def test_handle(self):
        self.no_change_pert.is_fuzzed = False
        Config.stage = Const.FORWARD

        params = HandlerParams()
        input = [Tensor([1.0], dtype=ms.float32), Tensor([5.0], dtype=ms.float32)]
        params.args = input
        params.kwargs = {}
        params.index = 0
        params.original_func = ops.add
        target = Tensor([6.0], dtype=ms.float32)
        ret = self.no_change_pert.handle(params)
        self.assertTrue(self.no_change_pert.is_fuzzed)
        self.assertTrue((params.fuzzed_value == input[0]).all())
        self.assertTrue((ret == target).all())

    @classmethod
    def tearDownClass(cls):
        cls.no_change_pert = None
