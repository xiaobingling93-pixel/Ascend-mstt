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
from mindspore import Tensor, ops

from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.common.utils import Tools
from msprobe.mindspore.free_benchmark.handler.base_handler import BaseHandler
from msprobe.mindspore.dump.hook_cell.api_register import get_api_register


class Handler(BaseHandler):
    def handle(self, params: HandlerParams):
        pass


def where(*args):
    return args[1]


def abs(tensor):
    return tensor


class TestBaseHandler(unittest.TestCase):
    base_handler = None

    @classmethod
    def setUpClass(cls):
        cls.base_handler = Handler("api_name_with_id")
        get_api_register(True).restore_all_api()

    def test___init__(self):
        base_handler = Handler("api_name_with_id")
        self.assertEqual(base_handler.api_name_with_id, "api_name_with_id")

    def test_pre_calculate(self):
        fuzzed_output = Tensor([1.0], dtype=ms.float32)
        original_output = Tensor([1.0], dtype=ms.float16)
        target = (original_output.to(fuzzed_output.dtype), fuzzed_output,
                  FreeBenchmarkConst.PERT_VALUE_DICT.get(ms.float32))
        ret = self.base_handler.pre_calculate(original_output, fuzzed_output)
        self.assertTrue((ret[0] == target[0]).all())
        self.assertEqual(ret[0].dtype, target[0].dtype)
        self.assertTrue((ret[1] == target[1]).all())
        self.assertEqual(ret[2], target[2])

    def test_get_threshold(self):
        target = Tools.get_default_error_threshold(ms.float32)
        ret = self.base_handler.get_threshold(ms.float32)
        self.assertEqual(ret, target)

    def test_convert_overflow_ratio_to_consistent(self):
        ratio = float("nan")
        target = FreeBenchmarkConst.NO_CHANGE_ERROR_THRESHOLD
        ret = self.base_handler.convert_overflow_ratio_to_consistent(ratio)
        self.assertEqual(ret, target)

        ratio = float("inf")
        ret = self.base_handler.convert_overflow_ratio_to_consistent(ratio)
        self.assertEqual(ret, target)

        ratio = 0.0001
        target = ratio
        ret = self.base_handler.convert_overflow_ratio_to_consistent(ratio)
        self.assertEqual(ret, target)

    def test_get_endless_norm(self):
        first_tensor = Tensor([1.0, 1.2], dtype=ms.float32)
        second_tensor = Tensor([1.5, 2.0], dtype=ms.float32)
        abs_tol = FreeBenchmarkConst.PERT_VALUE_DICT.get(ms.float32)
        target = ops.max(ops.div(second_tensor, first_tensor))[0].item()
        with patch.object(ops, "where", new=where), \
             patch.object(ops, "abs", new=abs):
            ret = self.base_handler.get_endless_norm(first_tensor, second_tensor, abs_tol)
            self.assertEqual(ret, target)

            first_tensor = Tensor([1.0, 1.2], dtype=ms.bfloat16)
            second_tensor = Tensor([1.5, 2.0], dtype=ms.bfloat16)
            target = ops.max(ops.div(ops.cast(second_tensor, ms.float32), ops.cast(first_tensor, ms.float32)))[0].item()
            ret = self.base_handler.get_endless_norm(first_tensor, second_tensor, abs_tol)
            self.assertEqual(ret, target)

    def test_ratio_calculate(self):
        original_output = Tensor([1.0, 1.2], dtype=ms.float32)
        fuzzed_output = Tensor([1.5, 2.0], dtype=ms.float32)
        target = ops.max(ops.div(fuzzed_output, original_output))[0].item()
        with patch.object(ops, "where", new=where), \
             patch.object(ops, "abs", new=abs):
            ret = self.base_handler.ratio_calculate(original_output, fuzzed_output)
        self.assertEqual(ret, target)

    @patch.object(logger, "error")
    def test_npu_compare(self, mock_error):
        original_output = 1.0
        fuzzed_output = 1.5
        target = (True, 1.0)
        ret = self.base_handler.npu_compare(original_output, fuzzed_output)
        mock_error.assert_called_with("The compare for output type `<class 'float'>` is not supported")
        self.assertEqual(ret, target)

        original_output = Tensor([1.0, 1.2], dtype=ms.float32)
        fuzzed_output = Tensor([1.5, 2.0], dtype=ms.float32)
        ratio = ops.max(ops.div(fuzzed_output, original_output))[0].item()
        target = (False, ratio)
        with patch.object(ops, "where", new=where), \
             patch.object(ops, "abs", new=abs):
            ret = self.base_handler.npu_compare(original_output, fuzzed_output)
        self.assertEqual(ret, target)

    def test_is_float_tensor(self):
        output = Tensor([1.0, 1.2], dtype=ms.float32)
        ret = self.base_handler.is_float_tensor(output)
        self.assertTrue(ret)

        output = [1.0, Tensor([1.0, 1.2], dtype=ms.float32)]
        ret = self.base_handler.is_float_tensor(output)
        self.assertTrue(ret)

        output = Tensor([1], dtype=ms.int32)
        ret = self.base_handler.is_float_tensor(output)
        self.assertFalse(ret)

    def test_handle(self):
        params = HandlerParams()
        ret = self.base_handler.handle(params)
        self.assertIsNone(ret)

    @classmethod
    def tearDownClass(cls):
        cls.base_handler = None
