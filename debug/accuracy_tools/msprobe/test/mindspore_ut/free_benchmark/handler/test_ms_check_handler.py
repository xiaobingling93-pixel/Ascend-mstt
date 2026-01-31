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
from mindspore import Tensor, ops

from msprobe.core.data_dump.json_writer import DataWriter
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.handler.check_handler import CheckHandler
from msprobe.core.common.runtime import Runtime


def where(*args):
    return args[1]


def abs(tensor):
    return tensor


class TestCheckHandler(unittest.TestCase):
    check_handler = None

    @classmethod
    def setUpClass(cls):
        cls.check_handler = CheckHandler("api_name_with_id")

    @classmethod
    def tearDownClass(cls):
        cls.check_handler = None

    @patch.object(CheckHandler, "npu_compare_and_save")
    @patch.object(logger, "error")
    def test_handle(self, mock_error, mock_compare):
        params = HandlerParams()
        params.original_result = Tensor([1.0], dtype=ms.float32)
        params.fuzzed_result = Tensor([1], dtype=ms.int32)
        self.check_handler.handle(params)
        mock_compare.assert_not_called()

        params.fuzzed_result = Tensor([1.0001], dtype=ms.float32)
        with patch.object(CheckHandler, "npu_compare_and_save") as mock_compare:
            self.check_handler.handle(params)
        mock_compare.assert_called_with(params.original_result, params.fuzzed_result, params)

        params.original_result = (Tensor([1.0], dtype=ms.float32), Tensor([2.0], dtype=ms.float32))
        params.fuzzed_result = (Tensor([1.0001], dtype=ms.float32), Tensor([2.0001], dtype=ms.float32))
        with patch.object(CheckHandler, "npu_compare_and_save") as mock_compare:
            self.check_handler.handle(params)
        self.assertEqual(mock_compare.call_count, 2)
        self.assertEqual(mock_compare.call_args_list[0][0],
                         (params.original_result[0], params.fuzzed_result[0], params))
        self.assertEqual(mock_compare.call_args_list[0][1], {"output_index": 0})
        self.assertEqual(mock_compare.call_args_list[1][0],
                         (params.original_result[1], params.fuzzed_result[1], params))
        self.assertEqual(mock_compare.call_args_list[1][1], {"output_index": 1})

        mock_error.assert_not_called()

    @patch.object(logger, "error")
    def test_npu_compare_and_save(self, mock_error):
        original_output = Tensor([1.0, 1.2], dtype=ms.float32)
        fuzzed_output = Tensor([1.5, 2.0], dtype=ms.float32)
        params = HandlerParams()
        params.original_result = original_output
        params.fuzzed_result = fuzzed_output
        data_dict = {
            "rank": None if Runtime.rank_id == -1 else Runtime.rank_id,
            "pert_type": Config.pert_type,
            "stage": Config.stage,
            "step": Runtime.step_count,
            "api_name": "api_name_with_id",
            "max_rel": ops.max(ops.div(fuzzed_output, original_output))[0].item() - 1,
            "dtype": ms.float32,
            "shape": original_output.shape,
            "output_index": None
        }
        with patch.object(DataWriter, "write_data_to_csv") as mock_write, \
             patch.object(ops, "where", new=where), \
             patch.object(ops, "abs", new=abs):
            self.check_handler.npu_compare_and_save(original_output, fuzzed_output, params)
        self.assertEqual(list(mock_write.call_args[0][0]), list(data_dict.values()))
        self.assertEqual(mock_write.call_args[0][1], data_dict.keys())
        self.assertEqual(mock_write.call_args[0][2], Config.dump_path)
        mock_error.assert_called_with("api_name_with_id is not consistent")
