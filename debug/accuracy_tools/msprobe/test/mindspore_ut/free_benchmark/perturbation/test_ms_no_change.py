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
