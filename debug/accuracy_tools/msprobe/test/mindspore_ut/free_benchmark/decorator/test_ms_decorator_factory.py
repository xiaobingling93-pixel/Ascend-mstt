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

import os
from unittest import TestCase
from unittest.mock import patch

import mindspore as ms
from mindspore import Tensor, ops

from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.decorator.decorator_factory import (data_pre_deal, decorate,
                                                                          decorate_forward_function,
                                                                          get_target_arg_index,
                                                                          need_wrapper_func, stack_depth_check)
from msprobe.mindspore.runtime import Runtime


class TestDecoratorFactory(TestCase):
    def test_need_wrapper_func(self):
        Runtime.is_running = True
        Config.is_enable = False
        self.assertFalse(need_wrapper_func())

        Runtime.is_running = False
        Config.is_enable = True
        self.assertFalse(need_wrapper_func())

        Runtime.is_running = True
        Config.is_enable = True
        with patch("msprobe.mindspore.free_benchmark.decorator.decorator_factory.stack_depth_check",
                   return_value=False):
            self.assertFalse(need_wrapper_func())

        with patch("msprobe.mindspore.free_benchmark.decorator.decorator_factory.stack_depth_check",
                   return_value=True):
            Config.steps = [0]
            Runtime.step_count = 1
            self.assertFalse(need_wrapper_func())

            Config.steps = []
            Config.ranks = [0]
            Runtime.rank_id = 1
            self.assertFalse(need_wrapper_func())

            Config.ranks = []
            self.assertTrue(need_wrapper_func())

            Config.ranks = [1]
            self.assertTrue(need_wrapper_func())

    def test_data_pre_deal(self):
        params = HandlerParams()
        params.args = (Tensor([1.0, 1.0], dtype=ms.float32), 1)
        params.kwargs = {"axis": 0}
        params.original_func = ops.split
        params.index = 0

        ret = data_pre_deal("ops.split", params.original_func, *params.args, **params.kwargs)
        self.assertTrue((ret.args[0] == params.args[0]).all())
        self.assertEqual(ret.args[1], params.args[1])
        self.assertEqual(ret.kwargs, params.kwargs)
        self.assertEqual(ret.original_func, params.original_func)
        self.assertEqual(ret.index, params.index)

        params.args = (Tensor([1, 1], dtype=ms.int32), 1)
        with self.assertRaises(Exception) as context:
            ret = data_pre_deal("ops.split", params.original_func, *params.args, **params.kwargs)
            self.assertEqual(str(context.exception), "ops.split has no supported input type")
            self.assertEqual(ret.index, -1)

    def test_get_target_arg_index(self):
        args = (1, Tensor([1.0, 1.0], dtype=ms.float32))
        target = 1
        ret = get_target_arg_index(args)
        self.assertEqual(ret, target)

        args = (1, (1.0, 1.0))
        target = 1
        ret = get_target_arg_index(args)
        self.assertEqual(ret, target)

        args = (1, Tensor([1.0, 1.0], dtype=ms.int32))
        target = -1
        ret = get_target_arg_index(args)
        self.assertEqual(ret, target)

    def test_stack_depth_check(self):
        def fuzz_wrapper(call_times):
            call_times += 1
            if stack_depth_check():
                call_times = fuzz_wrapper(call_times)
            return call_times

        ret = fuzz_wrapper(0)
        self.assertEqual(ret, 2)

    def test_decorate_forward_function(self):
        def func():
            pass

        with patch("msprobe.mindspore.free_benchmark.decorator.decorator_factory.decorate",
                   return_value=0) as mock_decorate:
            decorate_forward_function(func)
            ret = decorate_forward_function(func, api_name="api_name")
            self.assertEqual(mock_decorate.call_args_list[0][0][0], func)
            self.assertEqual(mock_decorate.call_args_list[0][0][1].__name__, "forward_func")
            self.assertEqual(mock_decorate.call_args_list[0][0][2], "func")
            self.assertEqual(mock_decorate.call_args_list[1][0][2], "api_name")
            self.assertEqual(ret, 0)

    def test_decorate(self):
        def decorate_func(input):
            if isinstance(input, int):
                return input + 1
            else:
                raise

        original_func = ops.add
        api_name = "ops.add"
        fuzz_wrapper = decorate(original_func, decorate_func, api_name)

        with patch("msprobe.mindspore.free_benchmark.decorator.decorator_factory.data_pre_deal",
                   return_value=0) as mock_pre_deal:
            args = (Tensor([1.0], dtype=ms.float32), Tensor([5.0], dtype=ms.float32))
            kwargs = {}
            os.environ["RANK_ID"] = "1"

            Runtime.rank_id = 0
            with patch("msprobe.mindspore.free_benchmark.decorator.decorator_factory.need_wrapper_func",
                       return_value=True), \
                 patch.object(logger, "info") as mock_info:
                ret = fuzz_wrapper(*args, **kwargs)
                mock_info.assert_called_with(f"[{api_name}] is checking.")
                mock_pre_deal.assert_called_with(api_name, original_func, *args, **kwargs)
                self.assertEqual(ret, 1)
                self.assertEqual(Runtime.rank_id, 0)

            Runtime.rank_id = -1
            with patch("msprobe.mindspore.free_benchmark.decorator.decorator_factory.need_wrapper_func",
                       return_value=False), \
                 patch.object(logger, "info") as mock_info:
                target = Tensor([6.0], dtype=ms.float32)
                ret = fuzz_wrapper(*args, **kwargs)
                mock_pre_deal.assert_called_once()
                self.assertEqual(ret, target)
                self.assertEqual(Runtime.rank_id, "1")

            del os.environ["RANK_ID"]

        with patch("msprobe.mindspore.free_benchmark.decorator.decorator_factory.data_pre_deal",
                   return_value="0") as mock_pre_deal:
            args = (Tensor([1.0], dtype=ms.float32), Tensor([5.0], dtype=ms.float32))
            kwargs = {}
            Runtime.rank_id = 0
            with patch("msprobe.mindspore.free_benchmark.decorator.decorator_factory.need_wrapper_func",
                       return_value=True), \
                 patch.object(logger, "info"), \
                 patch.object(logger, "error") as mock_error:
                target = Tensor([6.0], dtype=ms.float32)
                ret = fuzz_wrapper(*args, **kwargs)
                self.assertEqual(mock_error.call_count, 2)
                self.assertEqual(ret, target)
