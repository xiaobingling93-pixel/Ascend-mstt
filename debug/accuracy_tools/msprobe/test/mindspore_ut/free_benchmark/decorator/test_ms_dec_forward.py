# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
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

from unittest import TestCase
from unittest.mock import patch

import mindspore as ms
from mindspore import Tensor, ops

from msprobe.mindspore.common.const import Const, FreeBenchmarkConst
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.decorator.dec_forward import ForwardSelfChecker
from msprobe.mindspore.free_benchmark.handler.check_handler import CheckHandler


class TestForwardSelfChecker(TestCase):
    checker = None

    @classmethod
    def setUpClass(cls):
        cls.checker = ForwardSelfChecker("api_name")

    def test__init__(self):
        self_checker = ForwardSelfChecker("api_name")
        self.assertEqual(self_checker.api_name, "api_name")

    def test_get_compare_data(self):
        params = HandlerParams()
        params.args = (Tensor([1.0], dtype=ms.float32), Tensor([5.0], dtype=ms.float32))
        params.index = 0
        params.fuzzed_value = Tensor([1.0001], dtype=ms.float32)
        params.original_result = (Tensor([2.0], dtype=ms.float32), Tensor([6.0], dtype=ms.float32))
        params.fuzzed_result = (Tensor([2.0001], dtype=ms.float32), Tensor([6.0001], dtype=ms.float32))

        TestForwardSelfChecker.checker.api_name = "api_name"
        self.checker.get_compare_data(params)
        target = (Tensor([2.0001], dtype=ms.float32), Tensor([6.0001], dtype=ms.float32))
        self.assertTrue((params.fuzzed_result[0] == target[0]).all())
        self.assertTrue((params.fuzzed_result[1] == target[1]).all())

        TestForwardSelfChecker.checker.api_name = Const.COMMUNICATION_API_LIST[0]
        Config.pert_type = FreeBenchmarkConst.IMPROVE_PRECISION
        self.checker.get_compare_data(params)
        target = Tensor([1.0001], dtype=ms.float32)
        self.assertTrue((params.fuzzed_result == target).all())
        target = (Tensor([1.0], dtype=ms.float32), Tensor([5.0], dtype=ms.float32))
        self.assertTrue((params.original_result[0] == target[0]).all())
        self.assertTrue((params.original_result[1] == target[1]).all())

        params.fuzzed_value = Tensor([1.0001], dtype=ms.float32)
        params.original_result = (Tensor([2.0], dtype=ms.float32), Tensor([6.0], dtype=ms.float32))
        Config.pert_type = FreeBenchmarkConst.ADD_NOISE
        self.checker.get_compare_data(params)
        target = Tensor([1.0001], dtype=ms.float32)
        self.assertTrue((params.fuzzed_result == target).all())
        target = Tensor([1.0], dtype=ms.float32)
        self.assertTrue((params.original_result == target).all())

    def test_deal_fuzzed_and_original_result(self):
        params = HandlerParams()
        params.fuzzed_value = [Tensor([1.0001], dtype=ms.float32)]
        params.args = [Tensor([1.0], dtype=ms.float32)]
        params.original_result = Tensor([1.0], dtype=ms.float32)
        Config.handler_type = FreeBenchmarkConst.CHECK
        handler_return = Tensor([2.0], dtype=ms.float32)

        with patch.object(CheckHandler, "handle", return_value=handler_return) as mock_handle:
            TestForwardSelfChecker.checker.api_name = "api_name"
            ret = self.checker.deal_fuzzed_and_original_result(params)
            mock_handle.assert_called_with(params)
            self.assertTrue((ret == handler_return).all())

            TestForwardSelfChecker.checker.api_name = Const.COMMUNICATION_API_LIST[0]
            Config.pert_type = FreeBenchmarkConst.IMPROVE_PRECISION
            target = Tensor([1.0], dtype=ms.float32)
            ret = self.checker.deal_fuzzed_and_original_result(params)
            self.assertTrue((ret == target).all())

    def test_handle(self):
        params = HandlerParams()
        params.args = [Tensor([1.0], dtype=ms.float32), Tensor([5.0], dtype=ms.float32)]
        params.kwargs = {}
        params.index = 0
        params.original_func = ops.add
        original_result = ops.add(params.args[0], params.args[1])
        fuzzed_result = ops.add(original_result, 1e-8)
        deal_result = Tensor([2.0], dtype=ms.float32)

        with patch.object(ForwardSelfChecker,
                          "deal_fuzzed_and_original_result", return_value=deal_result) as mock_deal:
            Config.pert_type = FreeBenchmarkConst.ADD_NOISE
            ret = self.checker.handle(params)
            self.assertTrue((params.fuzzed_result == fuzzed_result).all())
            self.assertTrue((params.original_result == original_result).all())
            self.assertTrue((ret == deal_result).all())
            mock_deal.assert_called_with(params)

            params.args = [Tensor([0.0], dtype=ms.float32), Tensor([5.0], dtype=ms.float32)]
            ret = self.checker.handle(params)
            self.assertTrue((ret == params.original_result).all())
            mock_deal.assert_called_once()

    @classmethod
    def tearDownClass(cls):
        cls.checker = None
