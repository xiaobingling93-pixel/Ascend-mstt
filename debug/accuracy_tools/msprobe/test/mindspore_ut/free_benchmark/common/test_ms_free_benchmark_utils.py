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

import mindspore as ms
from mindspore import Tensor

from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.common.utils import Tools, UnequalRow, make_unequal_row
from msprobe.core.common.runtime import Runtime


class TestUtils(unittest.TestCase):
    def test_tools_get_first_tensor_dtype(self):
        tensor = Tensor([1.0], dtype=ms.float32)
        ret = Tools.get_first_tensor_dtype(tensor)
        self.assertEqual(ret, ms.float32)

        tensors = (Tensor([1.0], dtype=ms.float16), Tensor([5.0], dtype=ms.float32))
        ret = Tools.get_first_tensor_dtype(tensors)
        self.assertEqual(ret, ms.float16)

    def test_get_default_error_threshold(self):
        Config.pert_type = FreeBenchmarkConst.NO_CHANGE
        ret = Tools.get_default_error_threshold(None)
        self.assertEqual(ret, FreeBenchmarkConst.NO_CHANGE_ERROR_THRESHOLD)
        Config.pert_type = FreeBenchmarkConst.DEFAULT_PERT_TYPE

        ret = Tools.get_default_error_threshold(ms.float16)
        self.assertEqual(ret, FreeBenchmarkConst.ERROR_THRESHOLD.get(ms.float16))

    def test_get_grad_out(self):
        tensor = Tensor([1.0, 5.0], dtype=ms.float32)
        target_grad_out = Tensor([1.0, 1.0], dtype=ms.float32)
        ret = Tools.get_grad_out(tensor)
        self.assertTrue((ret == target_grad_out).all())

        tensors = (Tensor([1.0, 5.0], dtype=ms.float16), Tensor([1.0, 5.0], dtype=ms.float16))
        target_grad_out = (Tensor([1.0, 1.0], dtype=ms.float16), Tensor([1.0, 1.0], dtype=ms.float16))
        ret = Tools.get_grad_out(tensors)
        self.assertTrue((ret[0] == target_grad_out[0]).all())
        self.assertTrue((ret[1] == target_grad_out[1]).all())

    def test_unequal_row(self):
        self.assertIsNone(UnequalRow.rank)
        self.assertIsNone(UnequalRow.pert_type)
        self.assertIsNone(UnequalRow.stage)
        self.assertIsNone(UnequalRow.step)
        self.assertIsNone(UnequalRow.api_name)
        self.assertIsNone(UnequalRow.max_rel)
        self.assertIsNone(UnequalRow.dtype)
        self.assertIsNone(UnequalRow.shape)
        self.assertIsNone(UnequalRow.output_index)

    def test_make_unequal_row(self):
        api_name = "api_name"
        params = HandlerParams()
        params.original_result = (Tensor([1.0, 5.0], dtype=ms.float32), Tensor([1.0, 5.0], dtype=ms.float32))
        params.fuzzed_result = (Tensor([1.1, 5.0], dtype=ms.float32), Tensor([1.1, 5.0], dtype=ms.float32))
        target_row = UnequalRow(
            rank=Runtime.rank_id if Runtime.rank_id != -1 else None,
            api_name=api_name,
            pert_type=Config.pert_type,
            output_index=0,
            stage=Config.stage,
            step=Runtime.step_count,
            max_rel=1.003 - 1,
            dtype=params.original_result[0].dtype,
            shape=params.original_result[0].shape
        )
        ret = make_unequal_row(api_name, params, 1.003, 0)
        self.assertEqual(ret, target_row)
