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
