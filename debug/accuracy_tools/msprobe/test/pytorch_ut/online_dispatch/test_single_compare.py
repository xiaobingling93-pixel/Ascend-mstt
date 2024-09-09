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
from unittest.mock import Mock, patch
from msprobe.pytorch.online_dispatch.single_compare import SingleBenchmarkCompareStandard,SingleBenchmarkAccuracyResult, SingleBenchmarkAccuracyCompare

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
        self.single_benchmark_accuracy_result = SingleBenchmarkAccuracyResult(True,1,1,1,1,1)
    
    def test_get_result_result_false(self):
        self.single_benchmark_accuracy_result.get_result(0.5,0.5)
        self.assertEqual(self.single_benchmark_accuracy_result.result,False)

    def test_get_result_result_true(self):
        self.single_benchmark_accuracy_result.get_result(2,2)
        self.assertEqual(self.single_benchmark_accuracy_result.result,True)