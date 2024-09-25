# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
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

import math

import torch
from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.constant import ThresholdConfig
from msprobe.pytorch.free_benchmark.common.utils import TorchC


class SingleCompare:
    def __init__(self) -> None:
        self.relative_err = None
        self.absolute_err = None
        self.eb = None
        self.threshold = None

    @staticmethod
    def filter_overflow(tensor) -> int:
        inf_num = TorchC.sum(TorchC.isinf(tensor))
        nan_num = TorchC.sum(TorchC.isnan(tensor))
        return inf_num + nan_num

    @staticmethod
    def replace_inf_or_nan(tensor):
        finite_mask = TorchC.isfinite(tensor)
        inf_or_nan_mask = TorchC.logical_not(finite_mask)
        inf_or_nan_num = TorchC.sum(inf_or_nan_mask).item()
        if inf_or_nan_num > 0:
            tensor[inf_or_nan_mask] = 1
        return tensor

    @staticmethod
    def compare_float_seq(actual, golden):
        return math.isclose(actual, golden)

    @staticmethod
    def compare_other_seq(actual, golden):
        return actual == golden

    def compare_dict_seq(self, actual, golden):
        if len(actual) != len(golden):
            return False
        for key, value in golden.items():
            if not self.compare_seq(value, actual.get(key)):
                return False
        return True

    def compare_list_seq(self, actual, golden):
        if len(actual) != len(golden):
            return False
        for index_, value in enumerate(golden):
            if not self.compare_seq(value, actual[index_]):
                return False
        return True

    def compare_seq(self, actual, golden):
        if isinstance(golden, torch.Tensor):
            return self.compare_tensor_seq(actual, golden)
        elif isinstance(golden, dict):
            return self.compare_dict_seq(actual, golden)
        elif isinstance(golden, (tuple, list)):
            return self.compare_list_seq(actual, golden)
        elif isinstance(golden, float):
            return self.compare_float_seq(actual, golden)
        else:
            return self.compare_other_seq(actual, golden)

    def compare_tensor_seq(self, actual, golden):
        self.threshold = ThresholdConfig.BENCHMARK_THD_DICT.get(
            actual.dtype, ThresholdConfig.BENCHMARK_THD_DICT.get(torch.float32)
        )
        if self.filter_overflow(golden) > 0:
            logger.warning_on_rank_0("[msprobe] Free Benchmark: inf and nan"
                                  "in golden tensor is not supported.")
            return True
        actual = self.replace_inf_or_nan(actual)
        actual = actual.to(torch.float64)
        golden = golden.to(torch.float64).to(actual.device)
        self._cal_compare_metrics(actual, golden)
        if self.absolute_err > self.threshold.small_value_atol:
            return False
        if self.relative_err > self.threshold.rtol:
            return False
        if self.eb > self.threshold.err_balance:
            return False
        return True

    def _cal_compare_metrics(self, actual, golden):
        diff_value = TorchC.subtract(actual, golden)
        diff_abs = TorchC.abs(diff_value)
        golden_abs = TorchC.abs(golden)
        # 使用绝对误差的元素
        self.absolute_err = TorchC.max(TorchC.where(
            TorchC.lt(TorchC.abs(actual), self.threshold.small_value), diff_abs, 0
        ))
        diff_rel = TorchC.div(diff_abs, golden_abs)
        # 使用相对误差的元素
        self.relative_err = TorchC.max(TorchC.where(
            TorchC.ge(TorchC.abs(actual), self.threshold.small_value), diff_rel, 0
        ))
        # 获取误差均衡性
        divided = TorchC.where(
            TorchC.ge(TorchC.abs(golden), self.threshold.small_value), golden_abs, 1
        )
        self.eb = TorchC.mean(TorchC.div(diff_value, divided))
