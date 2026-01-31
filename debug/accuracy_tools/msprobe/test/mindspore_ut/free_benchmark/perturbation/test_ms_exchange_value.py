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

from mindspore import Tensor

from msprobe.mindspore.free_benchmark.perturbation.exchange_value import ExchangeValuePerturbation


class TestExchangeValuePerturbation(unittest.TestCase):
    def test_exchange_value(self):
        perturbation = ExchangeValuePerturbation("api_name")

        original_tensor = Tensor(1.0)
        original_cache_tensor = Tensor(1.0)
        target_tensor = Tensor(1.0)
        final_tensor = perturbation.exchange_value(original_tensor)
        self.assertTrue((original_tensor == original_cache_tensor).all())
        self.assertTrue((final_tensor == target_tensor).all())

        perturbation.is_fuzzed = False
        original_tensor = Tensor([[1.0, 2.0, 3.0]])
        original_cache_tensor = Tensor([[1.0, 2.0, 3.0]])
        target_tensor = Tensor([[3.0, 2.0, 1.0]])
        final_tensor = perturbation.exchange_value(original_tensor)
        self.assertTrue((original_tensor == original_cache_tensor).all())
        self.assertTrue((final_tensor == target_tensor).all())

        perturbation.is_fuzzed = False
        original_tensor = Tensor([1.0, 2.0, 3.0])
        original_cache_tensor = Tensor([1.0, 2.0, 3.0])
        target_tensor = Tensor([3.0, 2.0, 1.0])
        final_tensor = perturbation.exchange_value(original_tensor)
        self.assertTrue((original_tensor == original_cache_tensor).all())
        self.assertTrue((final_tensor == target_tensor).all())

        perturbation.is_fuzzed = False
        original_tensor = Tensor([1.0, 2.0, 3.0])
        original_cache_tensor = Tensor([1.0, 2.0, 3.0])
        original_tensors = [original_tensor, original_tensor, original_tensor]
        original_cache_tensors = [original_cache_tensor, original_cache_tensor, original_cache_tensor]
        target_tensor = Tensor([3.0, 2.0, 1.0])
        target_tensors = [target_tensor, original_tensor, original_tensor]
        final_tensors = perturbation.exchange_value(original_tensors)
        self.assertTrue((original_tensors[0] == original_cache_tensors[0]).all())
        self.assertTrue((original_tensors[1] == original_cache_tensors[1]).all())
        self.assertTrue((original_tensors[2] == original_cache_tensors[2]).all())
        self.assertTrue((final_tensors[0] == target_tensors[0]).all())
        self.assertTrue((final_tensors[1] == target_tensors[1]).all())
        self.assertTrue((final_tensors[2] == target_tensors[2]).all())

        perturbation.is_fuzzed = False
        original_tensor = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        original_cache_tensor = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target_tensor = Tensor([[6.0, 2.0, 3.0], [4.0, 5.0, 1.0]])
        final_tensor = perturbation.exchange_value(original_tensor)
        self.assertTrue((original_tensor == original_cache_tensor).all())
        self.assertTrue((final_tensor == target_tensor).all())

        perturbation.is_fuzzed = False
        original_tensor = Tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]]])
        original_cache_tensor = Tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]]])
        target_tensor = Tensor([[[6.0, 7.0], [2.0, 3.0], [3.0, 4.0]], [[4.0, 5.0], [5.0, 6.0], [1.0, 2.0]]])
        final_tensor = perturbation.exchange_value(original_tensor)
        self.assertTrue((original_tensor == original_cache_tensor).all())
        self.assertTrue((final_tensor == target_tensor).all())

        perturbation.is_fuzzed = False
        original_tensor = {"value": Tensor([1.0, 2.0, 3.0])}
        target_tensor = {"value": Tensor([3.0, 2.0, 1.0])}
        final_tensor = perturbation.exchange_value(original_tensor)
        self.assertTrue((target_tensor.get("value") == final_tensor.get("value")).all())
