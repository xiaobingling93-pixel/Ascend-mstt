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
from unittest.mock import patch, MagicMock
import zlib

import mindspore as ms
from mindspore import Tensor
import numpy as np

from msprobe.core.data_dump.data_processor.base import BaseDataProcessor
from msprobe.core.data_dump.data_processor.mindspore_processor import (
    MindsporeDataProcessor,
    TensorDataProcessor,
    OverflowCheckDataProcessor
)


class TestMindsporeDataProcessor(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = MindsporeDataProcessor(self.config, self.data_writer)
        self.tensor = ms.Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))

    def test_get_md5_for_tensor(self):
        tensor = ms.Tensor([1.0, 2.0, 3.0], dtype=ms.bfloat16)
        expected_crc32 = zlib.crc32(np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes())
        expected_md5 = f"{expected_crc32:08x}"
        result = self.processor.get_md5_for_tensor(tensor)
        self.assertEqual(result, expected_md5)

    def test_analyze_builtin(self):
        test_slice = slice(1, 3, None)
        expected_result = {"type": "slice", "value": [1, 3, None]}
        result = self.processor._analyze_builtin(test_slice)
        self.assertEqual(result, expected_result)

        test_int = 42
        expected_result = {"type": "int", "value": 42}
        result = self.processor._analyze_builtin(test_int)
        self.assertEqual(result, expected_result)

    def test_get_stat_info_float(self):
        tensor = ms.Tensor([1.0, 2.0, 3.0])
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, 3.0)
        self.assertEqual(result.min, 1.0)
        self.assertEqual(result.mean, 2.0)
        self.assertEqual(result.norm, ms.ops.norm(tensor).item())

    def test_get_stat_info_int(self):
        tensor = ms.Tensor([1, 2, 3], dtype=ms.int32)
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, 3)
        self.assertEqual(result.min, 1)
        self.assertEqual(result.mean, 2)
        self.assertEqual(result.norm, ms.ops.norm(tensor).item())

    def test_get_stat_info_bool(self):
        tensor = ms.Tensor([True, False, True])
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, True)
        self.assertEqual(result.min, False)
        self.assertIsNone(result.mean)
        self.assertIsNone(result.norm)

    @patch.object(MindsporeDataProcessor, 'get_md5_for_tensor')
    def test__analyze_tensor(self, get_md5_for_tensor):
        get_md5_for_tensor.return_value = "test_md5"
        tensor = ms.Tensor(np.array([1, 2, 3], dtype=np.int32))
        self.config.summary_mode = 'md5'
        suffix = "test_tensor"
        expected_result = {
            'type': 'mindspore.Tensor',
            'dtype': 'Int32',
            'shape': (3,),
            'Max': 3,
            'Min': 1,
            'Mean': 2,
            'Norm': ms.ops.norm(tensor).item(),
            'md5': 'test_md5',
        }
        result = self.processor._analyze_tensor(tensor, suffix)
        self.assertEqual(result, expected_result)


class TestTensorDataProcessor(unittest.TestCase):

    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = TensorDataProcessor(self.config, self.data_writer)
        self.data_writer.dump_tensor_data_dir = "./dump_data"
        self.processor.current_api_or_module_name = "test_api"
        self.processor.api_data_category = "input"

    @patch('msprobe.core.data_dump.data_processor.mindspore_processor.save_tensor_as_npy')
    def test_analyze_tensor(self, mock_save):
        self.config.framework = "mindspore"
        tensor = ms.Tensor([1.0, 2.0, 3.0])
        suffix = 'suffix'
        result = self.processor._analyze_tensor(tensor, suffix)
        mock_save.assert_called_once()
        expected = {
            'type': 'mindspore.Tensor',
            'dtype': str(tensor.dtype),
            'shape': tensor.shape,
            'Max': 3.0,
            'Min': 1.0,
            'Mean': 2.0,
            'Norm': ms.ops.norm(tensor).item(),
            'data_name': 'test_api.input.suffix.npy'
        }
        self.assertEqual(expected, result)


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

    @patch("msprobe.core.data_dump.data_processor.mindspore_processor.save_tensor_as_npy")
    def test_maybe_save_overflow_data(self, mock_save):
        self.data_processor.has_overflow = True
        tensor1 = Tensor(1)
        tensor2 = Tensor(2)
        self.data_processor.cached_tensors_and_file_paths = {"tensor1": tensor1, "tensor2": tensor2}
        self.data_processor.maybe_save_overflow_data()
        self.assertEqual(mock_save.call_args_list[0][0],
                         (tensor1, "tensor1"))
        self.assertEqual(mock_save.call_args_list[1][0],
                         (tensor2, "tensor2"))

    def test_is_terminated(self):
        self.data_processor.overflow_nums = -1
        self.assertFalse(self.data_processor.is_terminated)
        self.data_processor.real_overflow_nums = 2
        self.data_processor.overflow_nums = 2
        self.assertTrue(self.data_processor.is_terminated)
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
