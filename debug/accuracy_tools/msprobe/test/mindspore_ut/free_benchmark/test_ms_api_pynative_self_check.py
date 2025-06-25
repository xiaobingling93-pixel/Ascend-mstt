# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import os
from unittest import TestCase
from unittest.mock import patch

import mindspore as ms
from mindspore import Tensor, mint, ops

from msprobe.core.common.const import Const
from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.api_pynative_self_check import (
    ApiPyNativeSelfCheck,
    check_all_tensor,
    check_self,
    data_pre_deal,
    deal_fuzzed_and_original_result,
    get_module,
    get_supported_ops,
    get_target_arg_index,
    need_wrapper_func,
    _api_register
)
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.common.utils import Tools
from msprobe.mindspore.free_benchmark.handler.check_handler import CheckHandler
from msprobe.mindspore.free_benchmark.handler.fix_handler import FixHandler
from msprobe.core.common.runtime import Runtime


class DebuggerConfig:
    handler_type = FreeBenchmarkConst.CHECK
    pert_type = FreeBenchmarkConst.IMPROVE_PRECISION
    stage = Const.FORWARD
    dump_level = FreeBenchmarkConst.DEFAULT_DUMP_LEVEL
    step = []
    rank = []
    dump_path = "/dump_path"
    list = []


class Cell:
    def __init__(self):
        self.input_kwargs = {}


class TestApiPyNativeSelfCheck(TestCase):
    checker = None

    @classmethod
    def setUpClass(cls):
        config = DebuggerConfig()
        config.list = []
        cls.checker = ApiPyNativeSelfCheck(config)

    def test___init__(self):
        self.assertTrue(Config.is_enable)
        self.assertEqual(Config.handler_type, DebuggerConfig.handler_type)
        self.assertEqual(Config.pert_type, DebuggerConfig.pert_type)
        self.assertEqual(Config.stage, DebuggerConfig.stage)
        self.assertEqual(Config.dump_level, DebuggerConfig.dump_level)
        self.assertEqual(Config.steps, DebuggerConfig.step)
        self.assertEqual(Config.ranks, DebuggerConfig.rank)
        self.assertEqual(Config.dump_path, os.path.join(DebuggerConfig.dump_path, "free_benchmark.csv"))
        target_api_list = get_supported_ops()
        self.assertEqual(self.checker.api_list, target_api_list)

        config = DebuggerConfig()
        config.list = ["mindspore.ops.add"]
        self_checker = ApiPyNativeSelfCheck(config)
        target_api_list = set(config.list)
        self.assertEqual(self_checker.api_list, target_api_list)
        target_ori_func = {"mindspore.ops.add": ops.add}
        self.assertEqual(self_checker.ori_func, target_ori_func)

    def test_handle(self):
        with patch.object(_api_register, "initialize_hook") as mock_init_hook, \
             patch.object(_api_register, "register_all_api") as mock_set_hook:
            self.checker.handle()
        mock_init_hook.assert_called_with(self.checker.build_hook)
        mock_set_hook.assert_called_once()

    def test_build_hook(self):
        hook_set = self.checker.build_hook("Functional.add.")

        cell = Cell()
        cell.msprobe_input_kwargs = {}

        with patch("msprobe.mindspore.free_benchmark.api_pynative_self_check.need_wrapper_func", return_value=False):
            self.assertIsNone(hook_set.forward_hook(cell, "input", "output"))

        cell = Cell()
        cell.msprobe_input_kwargs = {}
        self.checker.api_list = ["mindspore.ops.add"]
        self.checker.ori_func["mindspore.ops.add"] = "add"
        with patch("msprobe.mindspore.free_benchmark.api_pynative_self_check.need_wrapper_func", return_value=True), \
             patch("msprobe.mindspore.free_benchmark.api_pynative_self_check.check_self",
                   return_value="ret") as mock_check:
            ret = hook_set.forward_hook(cell, ("input",), ("output",))
        self.assertEqual(ret, "ret")
        mock_check.assert_called_with("Functional.add.0", ("output",), "add", "input")

        self.assertIsNone(hook_set.backward_hook("cell", "grad_input", "grad_output"))

    def test_store_original_func(self):
        self.checker.api_list = ["mindspore.ops.add"]
        self.checker.ori_func = {}
        target_ori_func = {"mindspore.ops.add": ops.add}
        self.checker.store_original_func()
        self.assertEqual(self.checker.ori_func, target_ori_func)

    def test_get_supported_ops(self):
        yaml_api = {
            "ops": ["add"],
            "Tensor": ["div"],
            "mint": ["mean"],
            "mint.nn.functional": ["relu"]
            }
        with patch("msprobe.mindspore.free_benchmark.api_pynative_self_check.load_yaml", return_value=yaml_api):
            api_list = get_supported_ops()
            target_list = []
            if hasattr(ops, "add"):
                target_list.append("mindspore.ops.add")
            if hasattr(Tensor, "div"):
                target_list.append("mindspore.Tensor.div")
            if hasattr(mint, "mean"):
                target_list.append("mindspore.mint.mean")
            if hasattr(mint.nn.functional, "relu"):
                target_list.append("mindspore.mint.nn.functional.relu")
            self.assertEqual(api_list, set(target_list))

    def test_get_module(self):
        module_obj, orig_func = get_module("mindspore.ops.add")
        self.assertEqual(module_obj, ops)
        self.assertEqual(orig_func, ops.add)

    @patch.object(logger, "warning")
    def test_check_self(self, mock_warning):
        api_name_with_id = "Functional.add.0"
        output = (ms.Tensor([2.0], dtype=ms.float16),)
        ori_func = ops.add
        args = (ms.Tensor([1.0]), 1.0)
        kwargs = {}

        Config.stage = Const.BACKWARD
        self.assertFalse(check_self(api_name_with_id, output, ori_func, *args, **kwargs))
        mock_warning.assert_called_with(f"{api_name_with_id} has non-tensor input or output.")

        mock_warning.reset_mock()
        Config.stage = Const.FORWARD
        with patch.object(logger, "info") as mock_info, \
             patch.object(_api_register, "restore_all_api") as mock_set_ori, \
             patch.object(_api_register, "register_all_api") as mock_set_hook, \
             patch("msprobe.mindspore.free_benchmark.api_pynative_self_check.deal_fuzzed_and_original_result",
                   return_value="ret"):
            args = (1.0, 1.0)
            ret = check_self(api_name_with_id, output, ori_func, *args, **kwargs)
            self.assertIsNone(ret)
            mock_warning.assert_called_once()
            mock_info.assert_not_called()

            Config.pert_type = FreeBenchmarkConst.IMPROVE_PRECISION
            args = (ms.Tensor([1.0], dtype=ms.float32), ms.Tensor([1.0], dtype=ms.float32))
            ret = check_self(api_name_with_id, output, ori_func, *args, **kwargs)
            mock_info.assert_called_with(f"[{api_name_with_id}] is {Config.handler_type}ing.")
            mock_set_ori.assert_called_once()
            mock_set_hook.assert_called_once()
            self.assertIsNone(ret)

            mock_set_hook.reset_mock()
            args = (ms.Tensor([1.0], dtype=ms.float16), ms.Tensor([1.0], dtype=ms.float16))
            ret = check_self(api_name_with_id, output, ori_func, *args, **kwargs)
            mock_set_hook.assert_called_once()
            self.assertEqual(ret, "ret")

            Config.stage = Const.BACKWARD
            mock_set_hook.reset_mock()
            with patch.object(Tools, "get_grad") as mock_grad:
                ret = check_self(api_name_with_id, output, ori_func, *args, **kwargs)
            self.assertEqual(mock_grad.call_count, 2)
            mock_set_hook.assert_called_once()
            self.assertEqual(ret, "ret")
            Config.stage = Const.FORWARD

    def test_check_all_tensor(self):
        inputs = ms.Tensor([1.0])
        self.assertTrue(check_all_tensor(inputs))

        inputs = (ms.Tensor([1.0]), ms.Tensor([2.0]))
        self.assertTrue(check_all_tensor(inputs))

        inputs = (ms.Tensor([1.0]), 2.0)
        self.assertFalse(check_all_tensor(inputs))

    def test_get_target_arg_index(self):
        args = (ms.Tensor([1], dtype=ms.int32), 2.0, ms.Tensor([1.0]))
        self.assertEqual(get_target_arg_index(args), 2)

        args = ((1.0, 2.0), ms.Tensor([1.0]))
        self.assertEqual(get_target_arg_index(args), 0)

        args = (1.0, 2.0)
        self.assertEqual(get_target_arg_index(args), -1)

    def test_data_pre_deal(self):
        params = HandlerParams()
        params.args = (Tensor([1.0, 1.0], dtype=ms.float32), 1)
        params.kwargs = {"axis": 0}
        params.original_func = ops.split
        params.index = 0

        ret = data_pre_deal("Functional.split.0", params.original_func, *params.args, **params.kwargs)
        self.assertTrue((ret.args[0] == params.args[0]).all())
        self.assertEqual(ret.args[1], params.args[1])
        self.assertEqual(ret.kwargs, params.kwargs)
        self.assertEqual(ret.original_func, params.original_func)
        self.assertEqual(ret.index, params.index)

        params.args = (Tensor([1, 1], dtype=ms.int32), 1)
        with patch.object(logger, "warning") as mock_warning:
            ret = data_pre_deal("Functional.split.0", params.original_func, *params.args, **params.kwargs)
            mock_warning.assert_called_with("Functional.split.0 has no supported input type.")
            self.assertEqual(ret.index, -1)

    def test_need_wrapper_func(self):
        Runtime.is_running = True
        Config.is_enable = False
        self.assertFalse(need_wrapper_func())

        Runtime.is_running = False
        Config.is_enable = True
        self.assertFalse(need_wrapper_func())

        Runtime.is_running = True
        Config.is_enable = True
        self.assertTrue(need_wrapper_func())

        Config.steps = [1]
        Runtime.step_count = 0
        self.assertFalse(need_wrapper_func())

        Config.steps = []
        Runtime.step_count = 0

        Config.ranks = []
        Runtime.rank_id = -1
        self.assertTrue(need_wrapper_func())

        with patch("msprobe.mindspore.free_benchmark.api_pynative_self_check.get_rank_if_initialized", return_value=0):
            self.assertTrue(need_wrapper_func())
        self.assertEqual(Runtime.rank_id, 0)

        Config.ranks = [0]
        Runtime.rank_id = 1
        self.assertFalse(need_wrapper_func())
        Config.ranks = []
        Runtime.rank_id = -1

    def test_deal_fuzzed_and_original_result(self):
        params = HandlerParams()

        Config.handler_type = FreeBenchmarkConst.FIX
        with patch.object(FixHandler, "handle") as mock_fix:
            deal_fuzzed_and_original_result("api_name_with_id", params)
        mock_fix.assert_called_with(params)

        Config.handler_type = FreeBenchmarkConst.CHECK
        with patch.object(CheckHandler, "handle") as mock_check:
            deal_fuzzed_and_original_result("api_name_with_id", params)
        mock_check.assert_called_with(params)
