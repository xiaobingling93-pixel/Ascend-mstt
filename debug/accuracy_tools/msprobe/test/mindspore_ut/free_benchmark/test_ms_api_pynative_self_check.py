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
from mindspore import Tensor, mint, ops
from mindspore.communication import comm_func

from msprobe.core.common.const import Const
from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.free_benchmark.api_pynative_self_check import (ApiPyNativeSelFCheck, get_decorate_func,
                                                                      get_module, get_supported_ops, get_wrapper_obj,
                                                                      hijack, is_func_support_decorate)
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.decorator.decorator_factory import decorate_forward_function


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
    def test___init__(self):
        config = DebuggerConfig()
        config.list = []
        self_checker = ApiPyNativeSelFCheck(config)

        self.assertTrue(Config.is_enable)
        self.assertEqual(Config.handler_type, config.handler_type)
        self.assertEqual(Config.pert_type, config.pert_type)
        self.assertEqual(Config.stage, config.stage)
        self.assertEqual(Config.dump_level, config.dump_level)
        self.assertEqual(Config.steps, config.step)
        self.assertEqual(Config.ranks, config.rank)
        self.assertEqual(Config.dump_path, os.path.join(config.dump_path, "free_benchmark.csv"))
        target_api_list = get_supported_ops()
        self.assertEqual(self_checker.api_list, target_api_list)

        config.list = ["mindspore.ops.add"]
        self_checker = ApiPyNativeSelFCheck(config)
        target_api_list = set(config.list)
        self.assertEqual(self_checker.api_list, target_api_list)

    def test_handle(self):
        config = DebuggerConfig()
        config.list = []
        self_checker = ApiPyNativeSelFCheck(config)

        with patch("msprobe.mindspore.free_benchmark.api_pynative_self_check.hijack") as mock_hijack:
            self_checker.handle()
        self.assertEqual(mock_hijack.call_count, len(self_checker.api_list))

    def test_get_supported_ops(self):
        yaml_api = {
            "ops": ["add"],
            "Tensor": ["div"],
            "mint": ["mean"],
            "mint.nn.functional": ["relu"],
            "communication": ["all_reduce"]}
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
            if hasattr(comm_func, "all_reduce"):
                target_list.append("mindspore.communication.comm_func.all_reduce")
            self.assertEqual(api_list, set(target_list))

    def test_get_decorate_func(self):
        ret = get_decorate_func()
        self.assertEqual(ret, decorate_forward_function)

    def test_is_func_support_decorate(self):
        ret = is_func_support_decorate(ops.Add)
        self.assertFalse(ret)

        ret = is_func_support_decorate(ops.__name__)
        self.assertFalse(ret)

        ret = is_func_support_decorate(ops.add)
        self.assertTrue(ret)

    def test_get_wrapper_obj(self):
        with patch("msprobe.mindspore.free_benchmark.api_pynative_self_check.decorate_forward_function",
                   return_value=0) as mock_dec:
            ret = get_wrapper_obj(ops.add, "ops.add")
            mock_dec.assert_called_with(ops.add, "ops.add")
            self.assertEqual(ret, 0)
            ret = get_wrapper_obj(ops.__name__, "ops.ops.__name__")
            mock_dec.assert_called_once()
            self.assertEqual(ret, ops.__name__)

    def test_get_module(self):
        module_obj, orig_func = get_module("mindspore.ops.add")
        self.assertEqual(module_obj, ops)
        self.assertEqual(orig_func, ops.add)

    def test_hijack(self):
        def wrapped_func():
            pass
        with patch("msprobe.mindspore.free_benchmark.api_pynative_self_check.get_wrapper_obj",
                   return_value=wrapped_func) as mock_dec:
            hijack(" ")
            mock_dec.assert_not_called()
            ori_func_backup = ops.add
            hijack("mindspore.ops.add")
            wrapped_ori_func = getattr(ops, "add")
            self.assertEqual(wrapped_ori_func, wrapped_func)
            setattr(ops, "add", ori_func_backup)
