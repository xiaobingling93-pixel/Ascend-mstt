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

from typing import Any

import numpy as np
from mindspore import Tensor, ops

from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.perturbation.base_perturbation import BasePerturbation


class BitNoisePerturbation(BasePerturbation):

    def add_bit_noise(self, inputs) -> Any:
        if isinstance(inputs, Tensor):
            bit_len_type = self._get_bit_len_type(inputs)
            if bit_len_type is not False:
                sub_normal_np = np.finfo(FreeBenchmarkConst.MS_NUMPY_DTYPE_DICT.get(inputs.dtype)).smallest_normal
                sub_normal = Tensor(sub_normal_np)
                noise_type = list(FreeBenchmarkConst.MS_NUMPY_DTYPE_DICT.keys())[
                             list(FreeBenchmarkConst.MS_NUMPY_DTYPE_DICT.values()).index(bit_len_type)]
                noise = ops.full(inputs.shape, 1, dtype=noise_type)
                input_np = inputs.asnumpy()
                input_np_int = input_np.view(bit_len_type)
                result = Tensor(input_np_int)
                result = ops.where(ops.abs(inputs) > sub_normal,
                                   ops.bitwise_xor(result, noise), result)
                result_np = result.asnumpy()
                result_np_float = result_np.view(FreeBenchmarkConst.MS_NUMPY_DTYPE_DICT.get(inputs.dtype))
                self.is_fuzzed = True
                return Tensor(result_np_float)

        if isinstance(inputs, dict):
            return {k: self.add_bit_noise(v) for k, v in inputs.items()}
        if isinstance(inputs, (tuple, list)):
            return type(inputs)([self.add_bit_noise(v) for v in inputs])
        return inputs

    def handle(self, params: HandlerParams) -> any:
        args = params.args
        params.fuzzed_value = self.add_bit_noise(params.args[params.index])
        if not self.is_fuzzed:
            logger.warning(f"{self.api_name} can not add bit noise.")
            return False
        params.args = args
        return self.get_fuzzed_result(params)

    def _get_bit_len_type(self, tensor):
        if self.is_fuzzed:
            return False
        if not isinstance(tensor, Tensor) or not ops.is_floating_point(tensor) or \
           tensor.numel() == 0:
            return False
        bit_len_type = FreeBenchmarkConst.PERT_BIT_DICT.get(tensor.dtype)
        if not bit_len_type:
            return False
        pert_value = FreeBenchmarkConst.PERT_VALUE_DICT.get(tensor.dtype)
        if not pert_value:
            return False
        max_val = ops.max(ops.abs(tensor))[0].item()
        if max_val < pert_value:
            return False
        return bit_len_type
