#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
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
from mindspore import Tensor, ops, mint
import numpy as np

from msprobe.core.common.const import Const
from msprobe.core.data_dump.data_processor.base import BaseDataProcessor
from msprobe.core.data_dump.data_processor.mindspore_processor import (
    MindsporeDataProcessor,
    TensorDataProcessor,
    OverflowCheckDataProcessor,
    KernelDumpDataProcessor,
)
from msprobe.mindspore.common.log import logger


def patch_norm(value):
    return ops.norm(value)


setattr(mint, "norm", patch_norm)


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
        self.config.async_dump = False
        tensor = ms.Tensor([1.0, 2.0, 3.0])
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, 3.0)
        self.assertEqual(result.min, 1.0)
        self.assertEqual(result.mean, 2.0)
        self.assertEqual(result.norm, ms.ops.norm(tensor).item())

    def test_get_stat_info_float_async(self):
        self.config.async_dump = True
        tensor = ms.tensor([1.0, 2.0, 3.0])
        result = self.processor.get_stat_info(tensor)
        result_max = result.max
        result_min = result.min
        result_mean = result.mean
        result_norm = result.norm

        self.assertEqual(result_max.item(), 3.0)
        self.assertEqual(result_min.item(), 1.0)
        self.assertEqual(result_mean.item(), 2.0)
        self.assertEqual(result_norm.item(), ms.ops.norm(tensor).item())

    def test_get_stat_info_int(self):
        self.config.async_dump = False
        tensor = ms.Tensor([1, 2, 3], dtype=ms.int32)
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, 3)
        self.assertEqual(result.min, 1)
        self.assertEqual(result.mean, 2)
        self.assertEqual(result.norm, ms.ops.norm(tensor).item())

    def test_get_stat_info_int_async(self):
        self.config.async_dump = True
        tensor = ms.tensor([1, 2, 3])
        result = self.processor.get_stat_info(tensor)

        result_max = result.max
        result_min = result.min

        self.assertEqual(result_max.item(), 3.0)
        self.assertEqual(result_min.item(), 1.0)

    def test_get_stat_info_bool(self):
        self.config.async_dump = False
        tensor = ms.Tensor([True, False, True])
        result = self.processor.get_stat_info(tensor)
        self.assertEqual(result.max, True)
        self.assertEqual(result.min, False)
        self.assertIsNone(result.mean)
        self.assertIsNone(result.norm)

    def test_get_stat_info_bool_async(self):
        self.config.async_dump = True
        tensor = ms.Tensor([True, False, True])
        result = self.processor.get_stat_info(tensor)

        result_max = result.max
        result_min = result.min

        self.assertEqual(result_max.item(), True)
        self.assertEqual(result_min.item(), False)

    @patch.object(MindsporeDataProcessor, 'get_md5_for_tensor')
    def test__analyze_tensor(self, get_md5_for_tensor):
        get_md5_for_tensor.return_value = "test_md5"
        tensor = ms.Tensor(np.array([1, 2, 3], dtype=np.int32))
        self.config.summary_mode = 'md5'
        self.config.async_dump = False
        suffix = "test_tensor"
        expected_result = {
            'type': 'mindspore.Tensor',
            'dtype': 'Int32',
            'shape': (3,)
        }
        result = self.processor._analyze_tensor(tensor, suffix)
        # 删除不必要的字段
        result.pop('tensor_stat_index', None)
        result.pop('md5_index', None)

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
        self.config.async_dump = False
        tensor = ms.Tensor([1.0, 2.0, 3.0])
        suffix = 'suffix'
        result = self.processor._analyze_tensor(tensor, suffix)
        mock_save.assert_called_once()
        expected = {
            'type': 'mindspore.Tensor',
            'dtype': str(tensor.dtype),
            'shape': tensor.shape,
            'data_name': 'test_api.input.suffix.npy'
        }
        result.pop('tensor_stat_index', None)
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

        # from unittest.mock import MagicMock

    def test__analyze_maybe_overflow_tensor(self):
        # Mock DataWriter 和相关方法
        self.data_processor.data_writer = MagicMock()

        tensor_json = {Const.TENSOR_STAT_INDEX: 1}  # 修正：添加正确的 tensor_stat_index

        # 模拟返回值
        self.data_processor.data_writer.get_buffer_values_max.return_value = 10
        self.data_processor.data_writer.get_buffer_values_min.return_value = -10

        self.data_processor.has_overflow = False
        # 调用函数并检查没有溢出
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertFalse(self.data_processor.has_overflow)

        self.data_processor.has_overflow = False
        # max 值为 -np.inf，应该触发溢出
        self.data_processor.data_writer.get_buffer_values_max.return_value = -np.inf
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertTrue(self.data_processor.has_overflow)

        self.data_processor.has_overflow = False
        # max 值为 np.inf，应该触发溢出
        self.data_processor.data_writer.get_buffer_values_max.return_value = np.inf
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertTrue(self.data_processor.has_overflow)

        self.data_processor.has_overflow = False
        # max 值为 np.nan，应该触发溢出
        self.data_processor.data_writer.get_buffer_values_max.return_value = np.nan
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertTrue(self.data_processor.has_overflow)

        self.data_processor.has_overflow = False
        # max 值为 0，不会触发溢出
        self.data_processor.data_writer.get_buffer_values_max.return_value = 0
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertFalse(self.data_processor.has_overflow)

        self.data_processor.has_overflow = False
        # min 值为 -np.inf，应该触发溢出
        self.data_processor.data_writer.get_buffer_values_min.return_value = -np.inf
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertTrue(self.data_processor.has_overflow)

        self.data_processor.has_overflow = False
        # min 值为 np.inf，应该触发溢出
        self.data_processor.data_writer.get_buffer_values_min.return_value = np.inf
        self.data_processor._analyze_maybe_overflow_tensor(tensor_json)
        self.assertTrue(self.data_processor.has_overflow)

        self.data_processor.has_overflow = False
        # min 值为 np.nan，应该触发溢出
        self.data_processor.data_writer.get_buffer_values_min.return_value = np.nan
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
            mock_warning.assert_called_with("tensor_stat_index does not exist in tensor_json.")
            mock_super.assert_called_with("tensor", "suffix")
            self.assertEqual(ret.get("Max"), None)
            self.assertEqual(ret.get("data_name"), "dump_data_name")

        with patch("msprobe.core.data_dump.data_processor.mindspore_processor.path_len_exceeds_limit",
                   return_value=True):
            self.data_processor._analyze_tensor("tensor", "suffix")
            mock_warning.assert_called_with("tensor_stat_index does not exist in tensor_json.")


class TestKernelDumpDataProcessor(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.data_writer = MagicMock()
        self.processor = KernelDumpDataProcessor(self.config, self.data_writer)

    @patch.object(logger, 'warning')
    def test_print_unsupported_log(self, mock_logger_warning):
        self.processor._print_unsupported_log("test_api_name")
        mock_logger_warning.assert_called_with("The kernel dump does not support the test_api_name API.")

    @patch('msprobe.core.data_dump.data_processor.mindspore_processor.KernelDumpDataProcessor.start_kernel_dump')
    @patch('msprobe.core.data_dump.data_processor.mindspore_processor.has_adump', new=True)
    def test_analyze_pre_forward_with_adump(self, mock_start_kernel_dump):
        self.processor.analyze_forward_input("test_api_name", None, None)
        mock_start_kernel_dump.assert_called_once()
        self.assertTrue(self.processor.enable_kernel_dump)

    @patch('msprobe.core.data_dump.data_processor.mindspore_processor.has_adump', new=False)
    @patch.object(logger, 'warning')
    def test_analyze_pre_forward_without_adump(self, mock_logger_warning):
        self.processor.enable_kernel_dump = True
        self.processor.analyze_forward_input("test_api_name", None, None)
        mock_logger_warning.assert_called_with(
            "The current msprobe package does not compile adump, and kernel dump cannot be used.")
        self.assertFalse(self.processor.enable_kernel_dump)

    @patch('msprobe.core.data_dump.data_processor.mindspore_processor.KernelDumpDataProcessor.stop_kernel_dump')
    @patch.object(logger, 'info')
    def test_analyze_forward_successfully(self, mock_logger_info, mock_stop_kernel_dump):
        self.processor.enable_kernel_dump = True
        self.processor.analyze_forward_output('test_api_name', None, None)
        self.assertFalse(self.processor.enable_kernel_dump)
        mock_stop_kernel_dump.assert_called_once()
        mock_logger_info.assert_called_with("The kernel data of test_api_name is dumped successfully.")

    @patch('msprobe.core.data_dump.data_processor.mindspore_processor.has_adump', new=True)
    @patch('msprobe.core.data_dump.data_processor.mindspore_processor.KernelDumpDataProcessor.start_kernel_dump')
    def test_analyze_pre_backward_with_adump(self, mock_start_kernel_dump):
        self.processor.enable_kernel_dump = True
        self.processor.analyze_backward_input("test_api_name", None, None)
        self.assertTrue(self.processor.enable_kernel_dump)
        mock_start_kernel_dump.assert_called_once()

    @patch('msprobe.core.data_dump.data_processor.mindspore_processor.has_adump', new=False)
    @patch.object(logger, 'warning')
    def test_analyze_pre_backward_without_adump(self, mock_logger_warning):
        self.processor.enable_kernel_dump = True
        self.processor.analyze_backward_input("test_api_name", None, None)
        self.assertFalse(self.processor.enable_kernel_dump)
        mock_logger_warning.assert_called_with(
            "The current msprobe package does not compile adump, and kernel dump cannot be used.")

    @patch('msprobe.core.data_dump.data_processor.mindspore_processor.KernelDumpDataProcessor.stop_kernel_dump')
    @patch.object(logger, 'info')
    def test_analyze_backward_successfully(self, mock_logger_info, mock_stop_kernel_dump):
        self.processor.enable_kernel_dump = True
        self.processor.analyze_backward('test_api_name', None, None)
        self.assertFalse(self.processor.enable_kernel_dump)
        mock_stop_kernel_dump.assert_called_once()
        mock_logger_info.assert_called_with("The kernel data of test_api_name is dumped successfully.")

    def test_reset_status(self):
        self.processor.enable_kernel_dump = False
        self.processor.reset_status()
        self.assertTrue(self.processor.enable_kernel_dump)
