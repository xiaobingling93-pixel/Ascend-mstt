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

from typing import Dict

import numpy as np
import torch
from msprobe.pytorch.free_benchmark.common.enums import FuzzThreshold
from msprobe.pytorch.free_benchmark.common.params import BenchmarkThd


class CommonField:
    DEVICE = "device"
    META = "meta"
    FUZZ_TENSOR = "fuzz_tensor"
    REQUIRES_GRAD = "requires_grad"
    HOLD_PLACE = "hold_place"
    DISTRIBUTED_OP = "torch.distributed"
    GRADSAVER = "grad_saver"


class ThresholdConfig:
    PERTURBATION_VALUE_DICT: Dict = {
        torch.bfloat16: FuzzThreshold.BF16_THD,
        torch.float16: FuzzThreshold.F16_THD,
        torch.float32: FuzzThreshold.F32_THD,
        torch.float64: FuzzThreshold.F64_THD,
    }

    ABS_TOL_VALUE_DICT: Dict = {
        torch.bfloat16: FuzzThreshold.BF16_THD,
        torch.float16: FuzzThreshold.F16_THD,
        torch.float32: FuzzThreshold.F32_THD,
        torch.float64: FuzzThreshold.F64_THD,
    }

    # bit翻转需要匹配到等长或更长的整型
    PERTURBATION_BIT_DICT = {
        torch.bfloat16: torch.int16,
        torch.float16: torch.int16,
        torch.float32: torch.int32,
        torch.float64: torch.int64,
    }

    # 输入噪声下界
    NOISE_INPUT_LOWER_BOUND = 1e-8
    COMP_CONSISTENT = 1.0
    COMP_NAN = np.nan
    SYMBOL_FLIPPING = "symbol_flipping"
    BACKWARD_OUTPUT_LOWER_BOUND = 1e-3
    SMALL_VALUE = 1.0
    # 预热初始阈值
    PREHEAT_INITIAL_THD = 2.05
    API_THD_STEP = 2.0

    DTYPE_PER_THD = {
        torch.float16: 1.002,
        torch.bfloat16: 1.004,
        torch.float32: 1.0002,
    }
    BENCHMARK_THD_DICT = {
        torch.float32: BenchmarkThd(2**-14, 1.0, 2**-14, 1e-4),
        torch.float16: BenchmarkThd(2**-11, 1.0, 2**-11, 1e-4),
        torch.bfloat16: BenchmarkThd(2**-8, 1.0, 2**-8, 1e-4),
    }

    TENSOR_SPLIT_MAX_CHUNK = 128


class PreheatConfig:
    IF_PREHEAT = "if_preheat"
    PREHEAT_STEP = "preheat_step"
    MAX_SAMPLE = "max_sample"
