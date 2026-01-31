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
from unittest.mock import patch

import mindspore as ms
from mindspore import Tensor

from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.handler.fix_handler import FixHandler


class TestFixHandler(unittest.TestCase):
    fix_handler = None

    @classmethod
    def setUpClass(cls):
        cls.fix_handler = FixHandler("api_name_with_id")

    def test__init__(self):
        fix_handler = FixHandler("api_name_with_id")
        self.assertEqual(fix_handler.api_name_with_id, "api_name_with_id")

    def test_use_fuzzed_result(self):
        original_result = 1.0
        fuzzed_result = 1.0001
        target = 1.0
        ret = self.fix_handler.use_fuzzed_result(original_result, fuzzed_result)
        self.assertTrue(ret == target)

        original_result = Tensor([1.0], dtype=ms.float16)
        fuzzed_result = Tensor([1.0001], dtype=ms.float32)
        target = Tensor([1.0001], dtype=ms.float16)
        ret = self.fix_handler.use_fuzzed_result(original_result, fuzzed_result)
        self.assertTrue((ret == target).all())
        self.assertEqual(ret.dtype, target.dtype)

        original_result = {"output": Tensor([1.0], dtype=ms.float16)}
        fuzzed_result = {"output": Tensor([1.0001], dtype=ms.float32)}
        target = {"output": Tensor([1.0001], dtype=ms.float16)}
        ret = self.fix_handler.use_fuzzed_result(original_result, fuzzed_result)
        self.assertTrue((ret.get("output") == target.get("output")).all())
        self.assertEqual(ret.get("output").dtype, target.get("output").dtype)

        original_result = [Tensor([1.0], dtype=ms.float16)]
        fuzzed_result = [Tensor([1.0001], dtype=ms.float32)]
        target = [Tensor([1.0001], dtype=ms.float16)]
        ret = self.fix_handler.use_fuzzed_result(original_result, fuzzed_result)
        self.assertTrue((ret[0] == target[0]).all())
        self.assertEqual(ret[0].dtype, target[0].dtype)

    def test_handle(self):
        params = HandlerParams()
        params.original_result = Tensor([1.0], dtype=ms.float16)
        params.fuzzed_result = Tensor([1.0001], dtype=ms.float32)
        with patch.object(FixHandler, "use_fuzzed_result", return_value=1.0001) as mock_use:
            ret = self.fix_handler.handle(params)
        mock_use.assert_called_with(params.original_result, params.fuzzed_result)
        self.assertTrue(ret == 1.0001)

        def mock_use_fuzzed_result(*args):
            raise Exception("raise Exception")

        with patch.object(FixHandler, "use_fuzzed_result", new=mock_use_fuzzed_result), \
             patch.object(logger, "error") as mock_error:
            ret = self.fix_handler.handle(params)
        self.assertEqual(mock_error.call_count, 2)
        self.assertEqual(mock_error.call_args_list[0][0][0], "api_name_with_id failed to fix.")
        self.assertEqual(mock_error.call_args_list[1][0][0], "raise Exception")
        self.assertTrue((ret == params.original_result).all())

    @classmethod
    def tearDownClass(cls):
        cls.fix_handler = None
