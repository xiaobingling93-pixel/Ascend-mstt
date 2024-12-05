# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
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
from unittest.mock import patch

import numpy as np
import mindspore as ms
from mindspore import Tensor, ops

from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.perturbation.bit_noise import BitNoisePerturbation


class TestBitNoisePerturbation(unittest.TestCase):

    bit_noise_pert = None

    @classmethod
    def setUpClass(cls):
        cls.bit_noise_pert = BitNoisePerturbation("Mint.add.0")

    def test__get_bit_len_type(self):
        input = Tensor([1.0], dtype=ms.float32)
        self.bit_noise_pert.is_fuzzed = True
        self.assertFalse(self.bit_noise_pert._get_bit_len_type(input))
        self.bit_noise_pert.is_fuzzed = False

        input = Tensor([1], dtype=ms.int32)
        self.assertFalse(self.bit_noise_pert._get_bit_len_type(input))

        input = Tensor([1.0], dtype=ms.bfloat16)
        self.assertFalse(self.bit_noise_pert._get_bit_len_type(input))

        input = Tensor([1e-9], dtype=ms.float32)
        self.assertFalse(self.bit_noise_pert._get_bit_len_type(input))

        input = Tensor([1.0], dtype=ms.float32)
        self.assertEqual(self.bit_noise_pert._get_bit_len_type(input), np.int32)

    def test_add_bit_noise(self):
        self.bit_noise_pert.is_fuzzed = False

        input = Tensor([1.0], dtype=ms.float32)
        target = Tensor([1.0000001], dtype=ms.float32)
        ret = self.bit_noise_pert.add_bit_noise(input)
        allowed_error = Tensor([0.00000001], dtype=ms.float32)
        self.assertTrue(ops.abs(target - ret) < allowed_error)
        self.assertTrue(self.bit_noise_pert.is_fuzzed)
        self.bit_noise_pert.is_fuzzed = False

        input = {"input": Tensor([1.0], dtype=ms.float32)}
        target = {"input": Tensor([1.0000001], dtype=ms.float32)}
        ret = self.bit_noise_pert.add_bit_noise(input)
        self.assertTrue(ops.abs(target.get("input") - ret.get("input")) < allowed_error)
        self.assertTrue(self.bit_noise_pert.is_fuzzed)
        self.bit_noise_pert.is_fuzzed = False

        input = [Tensor([1.0], dtype=ms.float32)]
        target = [Tensor([1.0000001], dtype=ms.float32)]
        ret = self.bit_noise_pert.add_bit_noise(input)
        self.assertTrue(ops.abs(target[0] - ret[0]) < allowed_error)
        self.assertTrue(self.bit_noise_pert.is_fuzzed)
        self.bit_noise_pert.is_fuzzed = False

        input = 1.0
        target = 1.0
        ret = self.bit_noise_pert.add_bit_noise(input)
        self.assertTrue(ret == target)
        self.assertFalse(self.bit_noise_pert.is_fuzzed)

    @patch.object(logger, "warning")
    def test_handle(self, mock_warning):
        self.bit_noise_pert.is_fuzzed = False

        params = HandlerParams()
        input = [Tensor([1], dtype=ms.int32)]
        params.args = input
        params.index = 0
        ret = self.bit_noise_pert.handle(params)
        mock_warning.assert_called_with("Mint.add.0 can not add bit noise.")
        self.assertFalse(self.bit_noise_pert.is_fuzzed)
        self.assertFalse(ret)

        input = [Tensor([1.0], dtype=ms.float32)]
        fuzzed_value = Tensor([1.0000001], dtype=ms.float32)
        allowed_error = Tensor([0.00000001], dtype=ms.float32)
        params.args = input
        with patch.object(BitNoisePerturbation, "get_fuzzed_result") as mock_get_result:
            self.bit_noise_pert.handle(params)
            mock_get_result.assert_called_with(params)
        self.assertTrue(self.bit_noise_pert.is_fuzzed)
        self.assertTrue(ops.abs(fuzzed_value - params.fuzzed_value) < allowed_error)

    @classmethod
    def tearDownClass(cls):
        cls.bit_noise_pert = None
