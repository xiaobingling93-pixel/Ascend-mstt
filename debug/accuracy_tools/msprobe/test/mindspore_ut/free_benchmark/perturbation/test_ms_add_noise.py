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

from msprobe.mindspore.free_benchmark.perturbation.add_noise import AddNoisePerturbation
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.common.log import logger


class TestAddNoisePerturbation(unittest.TestCase):

    add_noise_pert = None

    @classmethod
    def setUpClass(cls):
        cls.add_noise_pert = AddNoisePerturbation("Mint.add.0")

    def test__get_noise(self):
        input = Tensor([1.0], dtype=ms.float32)
        self.add_noise_pert.is_fuzzed = True
        self.assertFalse(self.add_noise_pert._get_noise(input))
        self.add_noise_pert.is_fuzzed = False

        input = Tensor([1], dtype=ms.int32)
        self.assertFalse(self.add_noise_pert._get_noise(input))

        input = Tensor([1.0], dtype=ms.float32)
        noise = Tensor([1e-8], dtype=ms.float32)
        ret = self.add_noise_pert._get_noise(input)
        self.assertEqual(self.add_noise_pert.perturbation_value, 1e-8)
        self.assertTrue((ret == noise).all())

    def test_add_noise(self):
        self.add_noise_pert.is_fuzzed = False

        input = Tensor([1.0], dtype=ms.float32)
        target = input.add(1e-8)
        ret = self.add_noise_pert.add_noise(input)
        self.assertTrue((ret == target).all())
        self.assertTrue(self.add_noise_pert.is_fuzzed)
        self.add_noise_pert.is_fuzzed = False

        input = {"input": Tensor([1.0], dtype=ms.float32)}
        target = {"input": input.get("input").add(1e-8)}
        ret = self.add_noise_pert.add_noise(input)
        self.assertTrue((ret.get("input") == target.get("input")).all())
        self.assertTrue(self.add_noise_pert.is_fuzzed)
        self.add_noise_pert.is_fuzzed = False

        input = [Tensor([1.0], dtype=ms.float32)]
        target = [input[0].add(1e-8)]
        ret = self.add_noise_pert.add_noise(input)
        self.assertTrue((ret[0] == target[0]).all())
        self.assertTrue(self.add_noise_pert.is_fuzzed)
        self.add_noise_pert.is_fuzzed = False

        input = 1.0
        target = 1.0
        ret = self.add_noise_pert.add_noise(input)
        self.assertTrue(ret == target)
        self.assertFalse(self.add_noise_pert.is_fuzzed)

    @patch.object(logger, "warning")
    def test_handle(self, mock_warning):
        self.add_noise_pert.is_fuzzed = False

        params = HandlerParams()
        input = [Tensor([1], dtype=ms.int32)]
        params.args = input
        params.index = 0
        ret = self.add_noise_pert.handle(params)
        mock_warning.assert_called_with("Mint.add.0 can not add noise.")
        self.assertFalse(self.add_noise_pert.is_fuzzed)
        self.assertFalse(ret)

        input = [Tensor([1.0], dtype=ms.float32)]
        fuzzed_value = input[0].add(1e-8)
        params.args = input
        with patch.object(AddNoisePerturbation, "get_fuzzed_result") as mock_get_result:
            self.add_noise_pert.handle(params)
            mock_get_result.assert_called_with(params)
        self.assertTrue(self.add_noise_pert.is_fuzzed)
        self.assertTrue((params.fuzzed_value == fuzzed_value).all())

    @classmethod
    def tearDownClass(cls):
        cls.add_noise_pert = None
