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

import os
from unittest import TestCase
from unittest.mock import patch

import mindspore as ms
from mindspore import Tensor, mint, ops

from msprobe.core.common.const import Const
from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.dump.hook_cell.api_registry import api_register
from msprobe.mindspore.free_benchmark.api_pynative_self_check import (ApiPyNativeSelFCheck, get_module,
                                                                      get_supported_ops, data_pre_deal)
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams


class DebuggerConfig:
    handler_type = FreeBenchmarkConst.CHECK
    pert_type = FreeBenchmarkConst.IMPROVE_PRECISION
    stage = Const.FORWARD
    dump_level = FreeBenchmarkConst.DEFAULT_DUMP_LEVEL
    step = []
    rank = []
    dump_path = "/dump_path"
    list = []


class TestApiPyNativeSelFCheck(TestCase):
    checker = None

    @classmethod
    def setUpClass(cls):
        config = DebuggerConfig()
        config.list = []
        cls.checker = ApiPyNativeSelFCheck(config)

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
        self_checker = ApiPyNativeSelFCheck(config)
        target_api_list = set(config.list)
        self.assertEqual(self_checker.api_list, target_api_list)
        target_ori_func = {"mindspore.ops.add": ops.add}
        self.assertEqual(self_checker.ori_func, target_ori_func)

    def test_handle(self):
        with patch.object(api_register, "initialize_hook") as mock_init_hook, \
             patch.object(api_register, "api_set_hook_func") as mock_set_hook:
            self.checker.handle()
        mock_init_hook.assert_called_with(self.checker.build_hook)
        mock_set_hook.assert_called_once()

    def test_store_original_func(self):
        self.checker.api_list = set(("mindspore.ops.add",))
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
