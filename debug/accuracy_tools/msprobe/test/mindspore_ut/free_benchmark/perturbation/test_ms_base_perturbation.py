# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
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

import unittest
from unittest.mock import patch

import mindspore as ms
from mindspore import Tensor
from msprobe.core.common.const import Const
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.perturbation.base_perturbation import BasePerturbation


class TestBasePerturbation(unittest.TestCase):
    base_pert = None

    def test___init__(self):
        TestBasePerturbation.base_pert = BasePerturbation("Functional.add.0")
        self.assertEqual(TestBasePerturbation.base_pert.api_name_with_id, "Functional.add.0")
        self.assertFalse(TestBasePerturbation.base_pert.is_fuzzed)
        self.assertIsNone(TestBasePerturbation.base_pert.perturbation_value)

    @patch("msprobe.core.hook_manager.BaseHookManager._should_execute_hook", return_value=False)
    def test_get_fuzzed_result(self, _):
        params = HandlerParams()
        params.args = [Tensor([1.0], dtype=ms.float32), Tensor([5.0], dtype=ms.float32)]
        params.kwargs = {}
        params.fuzzed_value = Tensor([2.0], dtype=ms.float32)
        params.index = 0
        params.original_func = ms.ops.add

        Config.stage = Const.BACKWARD
        target = (Tensor([1.0], dtype=ms.float32), Tensor([1.0], dtype=ms.float32))
        ret = self.base_pert.get_fuzzed_result(params)
        self.assertTrue((ret[0] == target[0]).all())
        self.assertTrue((ret[1] == target[1]).all())

        Config.stage = Const.FORWARD
        target = Tensor([7.0], dtype=ms.float32)
        ret = self.base_pert.get_fuzzed_result(params)
        self.assertTrue((ret == target).all())

    def handler(self):
        params = HandlerParams()
        ret = self.base_pert.handler(params)
        self.assertIsNone(ret)

    @classmethod
    def tearDownClass(cls):
        cls.base_pert = None
