#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
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
"""
import unittest
from unittest.mock import patch

from mindspore import Tensor
import numpy as np

from msprobe.core.data_dump.data_processor.base import BaseDataProcessor
from msprobe.core.data_dump.data_processor.mindspore_processor import MindsporeDataProcessor, OverflowCheckDataProcessor
from msprobe.core.common.const import FileCheckConst


class TestOverflowCheckDataProcessor(unittest.TestCase):
    def setUp(self):
        class Config:
            def __init__(self):
                self.overflow_nums = 1
        self.data_processor = OverflowCheckDataProcessor(Config(), None)

    def test___init__(self):
        self.assertEqual(self.data_processor.cached_tensors_and_file_paths, {})
        self.assertEqual(self.data_processor.real_overflow_nums, 0)
        self.assertEqual(self.data_processor.overflow_nums, 1)

    def test_analyze_forward(self):
        def func(_):
            self.data_processor.has_overflow = True
        with patch.object(BaseDataProcessor, "analyze_forward", return_value={"min", 0}):
            with patch.object(OverflowCheckDataProcessor, "maybe_save_overflow_data"):
                api_info = self.data_processor.analyze_forward("name", "module", "module_input_output")
            self.assertFalse(self.data_processor.has_overflow)
            self.assertIsNone(api_info)
            with patch.object(OverflowCheckDataProcessor, "maybe_save_overflow_data", new=func):
                api_info = self.data_processor.analyze_forward("name", "module", "module_input_output")
            self.assertTrue(self.data_processor.has_overflow)
            self.assertEqual(api_info, {"min", 0})

    def test_analyze_backward(self):
        def func(_):
            self.data_processor.has_overflow = True
        with patch.object(BaseDataProcessor, "analyze_backward", return_value={"min", 0}):
            with patch.object(OverflowCheckDataProcessor, "maybe_save_overflow_data"):
                api_info = self.data_processor.analyze_backward("name", "module", "module_input_output")
            self.assertFalse(self.data_processor.has_overflow)
            self.assertIsNone(api_info)
            with patch.object(OverflowCheckDataProcessor, "maybe_save_overflow_data", new=func):
                api_info = self.data_processor.analyze_backward("name", "module", "module_input_output")
            self.assertTrue(self.data_processor.has_overflow)
            self.assertEqual(api_info, {"min", 0})

    @patch("msprobe.core.data_dump.data_processor.mindspore_processor.np.save")
    @patch("msprobe.core.data_dump.data_processor.mindspore_processor.change_mode")
    def test_maybe_save_overflow_data(self, mock_change_mode, mock_save):
        self.data_processor.has_overflow = True
        tensor1 = Tensor(1)
        tensor2 = Tensor(2)
        self.data_processor.cached_tensors_and_file_paths = {"tensor1": tensor1, "tensor2": tensor2}
        with patch("mindspore.Tensor.asnumpy", return_value="npy"):
            self.data_processor.maybe_save_overflow_data()
        self.assertEqual(mock_save.call_args_list[0][0],
                         ("tensor1", "npy"))
        self.assertEqual(mock_save.call_args_list[1][0],
                         ("tensor2", "npy"))
        self.assertEqual(mock_change_mode.call_args_list[0][0],
                         ("tensor1", FileCheckConst.DATA_FILE_AUTHORITY))
        self.assertEqual(mock_change_mode.call_args_list[1][0],
                         ("tensor2", FileCheckConst.DATA_FILE_AUTHORITY))

    @patch("msprobe.core.data_dump.data_processor.mindspore_processor.logger.info")
    def test_is_terminated(self, mock_info):
        self.data_processor.overflow_nums = -1
        self.assertFalse(self.data_processor.is_terminated)
        self.data_processor.real_overflow_nums = 2
        self.data_processor.overflow_nums = 2
        self.assertTrue(self.data_processor.is_terminated)
        mock_info.assert_called_with("[msprobe] 超过预设溢出次数 当前溢出次数: 2")
        self.data_processor.overflow_nums = 3
        self.assertFalse(self.data_processor.is_terminated)

    def test__analyze_maybe_overflow_tensor(self):
        self.data_processor.has_overflow = False
        tensor_json = {"Max": None, "Min": 0}
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertFalse(self.data_processor.has_overflow)
        tensor_json.update({"Max": -np.inf})
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertTrue(self.data_processor.has_overflow)
        self.data_processor.has_overflow = False
        tensor_json.update({"Max": np.inf})
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertTrue(self.data_processor.has_overflow)
        self.data_processor.has_overflow = False
        tensor_json.update({"Max": np.nan})
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertTrue(self.data_processor.has_overflow)
        tensor_json.update({"Max": 0})
        self.data_processor.has_overflow = False
        tensor_json.update({"Min": -np.inf})
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertTrue(self.data_processor.has_overflow)
        self.data_processor.has_overflow = False
        tensor_json.update({"Min": np.inf})
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertTrue(self.data_processor.has_overflow)
        self.data_processor.has_overflow = False
        tensor_json.update({"Min": np.nan})
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertTrue(self.data_processor.has_overflow)

    @patch("msprobe.core.data_dump.data_processor.mindspore_processor.logger.warning")
    @patch.object(OverflowCheckDataProcessor, "get_save_file_path")
    @patch.object(MindsporeDataProcessor, "_analyze_tensor")
    def test__analyze_tensor(self, mock_super, mock_get_file_path, mock_warning):
        mock_get_file_path.return_value = ("dump_data_name", "file_path")
        single_arg = {"Max": None}
        mock_super.return_value = single_arg

        with patch("msprobe.core.data_dump.data_processor.mindspore_processor.path_len_exceeds_limit",
                   return_value=False):
            ret = self.data_processor._analyze_tensor("tensor", "suffix")
            self.assertEqual(self.data_processor.cached_tensors_and_file_paths, {"file_path": "tensor"})
            mock_warning.assert_not_called()
            mock_super.assert_called_with("tensor", "suffix")
            self.assertEqual(ret.get("Max"), None)
            self.assertEqual(ret.get("data_name"), "dump_data_name")

        with patch("msprobe.core.data_dump.data_processor.mindspore_processor.path_len_exceeds_limit",
                   return_value=True):
            self.data_processor._analyze_tensor("tensor", "suffix")
            mock_warning.assert_called_with("The file path file_path length exceeds limit.")
