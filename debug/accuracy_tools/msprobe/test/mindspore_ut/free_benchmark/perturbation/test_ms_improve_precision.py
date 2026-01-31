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

from msprobe.core.common.const import Const
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.perturbation.improve_precision import ImprovePrecisionPerturbation


class TestImprovePrecisionPerturbation(unittest.TestCase):

    improve_precision_pert = None

    @classmethod
    def setUpClass(cls):
        cls.improve_precision_pert = ImprovePrecisionPerturbation("Functional.add.0")

    def test_improve_tensor_precision(self):
        self.improve_precision_pert.is_fuzzed = False

        input = Tensor([1.0], dtype=ms.float16)
        target = Tensor([1.0], dtype=ms.float32)
        ret = self.improve_precision_pert.improve_tensor_precision(input)
        self.assertTrue((ret == target).all())
        self.assertEqual(ret.dtype, target.dtype)
        self.assertTrue(self.improve_precision_pert.is_fuzzed)
        self.improve_precision_pert.is_fuzzed = False

        input = {"input": Tensor([1.0], dtype=ms.float16)}
        target = {"input": Tensor([1.0], dtype=ms.float32)}
        ret = self.improve_precision_pert.improve_tensor_precision(input)
        self.assertTrue((ret.get("input") == target.get("input")).all())
        self.assertEqual(ret.get("input").dtype, target.get("input").dtype)
        self.assertTrue(self.improve_precision_pert.is_fuzzed)
        self.improve_precision_pert.is_fuzzed = False

        input = [Tensor([1.0], dtype=ms.float16)]
        target = [Tensor([1.0], dtype=ms.float32)]
        ret = self.improve_precision_pert.improve_tensor_precision(input)
        self.assertTrue((ret[0] == target[0]).all())
        self.assertEqual(ret[0].dtype, target[0].dtype)
        self.assertTrue(self.improve_precision_pert.is_fuzzed)
        self.improve_precision_pert.is_fuzzed = False

        input = Tensor([1.0], dtype=ms.float64)
        target = Tensor([1.0], dtype=ms.float64)
        ret = self.improve_precision_pert.improve_tensor_precision(input)
        self.assertTrue((ret == target).all())
        self.assertEqual(ret.dtype, target.dtype)
        self.assertFalse(self.improve_precision_pert.is_fuzzed)

    @patch("msprobe.core.hook_manager.BaseHookManager._should_execute_hook", return_value=False)
    @patch.object(logger, "warning")
    def test_handle(self, mock_warning, _):
        self.improve_precision_pert.is_fuzzed = False

        params = HandlerParams()
        input = [Tensor([1.0], dtype=ms.float32)]
        params.args = input
        params.kwargs = {}
        ret = self.improve_precision_pert.handle(params)
        mock_warning.assert_called_with("Functional.add.0 can not improve precision.")
        self.assertFalse(self.improve_precision_pert.is_fuzzed)
        self.assertFalse(ret)

        Config.stage = Const.FORWARD
        params.args = [Tensor([1.0], dtype=ms.float16), Tensor([5.0], dtype=ms.float16)]
        params.original_func = ms.ops.add
        target = Tensor([6.0], dtype=ms.float32)
        ret = self.improve_precision_pert.handle(params)
        self.assertTrue(ret == target)

        Config.stage = Const.BACKWARD
        params.args = [Tensor([1.0], dtype=ms.float16), Tensor([5.0], dtype=ms.float16)]
        params.original_func = ms.ops.add
        target = (Tensor([1.0], dtype=ms.float32), Tensor([1.0], dtype=ms.float32))
        ret = self.improve_precision_pert.handle(params)
        self.assertTrue(ret == target)

    @classmethod
    def tearDownClass(cls):
        cls.improve_precision_pert = None
