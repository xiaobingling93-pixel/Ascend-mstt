# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

import unittest
import torch
import logging
from msprobe.pytorch.common.log import logger
from unittest.mock import Mock, patch, MagicMock
from msprobe.pytorch.online_dispatch.single_compare import SingleBenchmarkCompareStandard, \
    SingleBenchmarkAccuracyResult, SingleBenchmarkAccuracyCompare


class TestSingleBenchmarkCompareStandard(unittest.TestCase):
    def setUp(self):
        self.single_benchmark_compare_standard = SingleBenchmarkCompareStandard()

    @patch('logging.warning')
    def test_get_error_thd_when_input_f64(self,mock_warning):
        self.single_benchmark_compare_standard.get_error_thd(torch.float64)
        mock_warning.assert_called_once_with("the output data of fp64 uses the same standard as fp32.")

    @patch('logging.error')
    def test_get_error_thd_when_input_bool(self,mock_error):
        self.single_benchmark_compare_standard.get_error_thd(torch.bool)
        mock_error.assert_called_once_with("Single benchmark compare only supports floating point "
            "in fp16, bf16, fp32. ")

    def test_get_eb_thd_when_input_f16(self):
        self.assertEqual(self.single_benchmark_compare_standard.get_eb_thd(torch.float16),2 ** -10)

    def test_get_eb_thd_when_input_bool(self):
        self.assertIsNone(self.single_benchmark_compare_standard.get_eb_thd(torch.bool))


class TestSingleBenchmarkAccuracyResult(unittest.TestCase):
    def setUp(self):
        self.single_benchmark_accuracy_result = SingleBenchmarkAccuracyResult(True, 1, 1,
                                                                              1, 1, 1)
    
    def test_get_result_result_false(self):
        self.single_benchmark_accuracy_result.get_result(0.5, 0.5)
        self.assertEqual(self.single_benchmark_accuracy_result.result, False)

    def test_get_result_result_true(self):
        self.single_benchmark_accuracy_result.get_result(2, 2)
        self.assertEqual(self.single_benchmark_accuracy_result.result, True)


class TestSingleBenchmarkAccuracyCompare(unittest.TestCase):
    @patch('logging.debug')
    @patch('logging.error')
    def test_check_output_size_pass(self, mock_error, mock_debug):
        # 模拟 npu_out 和 bench_out，它们的 numel() 返回 0
        npu_out = MagicMock()
        bench_out = MagicMock()
        npu_out.numel.return_value = 0
        bench_out.numel.return_value = 0
        npu_out.size.return_value = 0
        bench_out.size.return_value = 0

        result = SingleBenchmarkAccuracyCompare.check_output_size(npu_out, bench_out)

        self.assertTrue(result.result)
        mock_debug.assert_called_once_with(
            "The npu_output is [], and it is same as benchmark_output, "
            "the result of data_compare is Pass"
        )
        mock_error.assert_not_called()

    @patch('logging.debug')
    @patch('logging.error')
    def test_check_output_size_fail(self, mock_error, mock_debug):
        # 模拟 npu_out 和 bench_out，它们的 size() 不一样
        npu_out = MagicMock()
        bench_out = MagicMock()
        npu_out.numel.return_value = 10
        bench_out.numel.return_value = 10
        npu_out.size.return_value = [2, 3]
        bench_out.size.return_value = [3, 2]

        result = SingleBenchmarkAccuracyCompare.check_output_size(npu_out, bench_out)

        self.assertFalse(result.result)
        mock_error.assert_called_once_with(
            "the size of npu output[[2, 3]] and"
            "benchmark[[3, 2]] is not equal"
        )
        mock_debug.assert_not_called()

    def test_check_output_invalid_value_with_nan_and_inf(self):
        output = torch.tensor([float('nan'), float('inf')])
        result = SingleBenchmarkAccuracyCompare.check_output_invalid_value(output)

        self.assertTrue(result)

    def test_check_output_invalid_value_without_nan_or_inf(self):
        output = torch.tensor([1.0, 2.0, 3.0])
        result = SingleBenchmarkAccuracyCompare.check_output_invalid_value(output)

        self.assertFalse(result)

    @patch('msprobe.pytorch.common.log.logger.info')
    def test_compute_binary_diff(self, mock_info):
        npu_out = torch.Tensor([1, 2, 3, 4])
        bench_out = torch.Tensor([1, 2, 3, 4])
        result = SingleBenchmarkAccuracyCompare.compute_binary_diff(npu_out, bench_out)

        self.assertTrue(result.result)
        self.assertEqual(mock_info.call_count, 3)
