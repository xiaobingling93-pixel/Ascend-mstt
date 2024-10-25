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

from mindspore import Tensor, ops

from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.perturbation.base_perturbation import BasePerturbation


class AddNoisePerturbation(BasePerturbation):

    def handle(self, params: HandlerParams) -> Any:
        """
        返回增加扰动后的api输出

        """
        params.fuzzed_value = self.add_noise(params.args[params.index])
        if not self.is_fuzzed:
            logger.warning(f"{self.api_name} can not add noise.")
            return False
        return self.get_fuzzed_result(params)

    def add_noise(self, inputs) -> Any:
        """
        返回增加扰动后的api输入

        """
        if isinstance(inputs, Tensor):
            noise = self._get_noise(inputs)
            if noise is not False:
                result = ops.where(ops.abs(inputs) > self.perturbation_value ** 0.5,
                                   ops.add(noise, inputs), inputs)
                result = result.type(dtype=inputs.dtype)
                self.is_fuzzed = True
                return result

        if isinstance(inputs, dict):
            return {k: self.add_noise(v) for k, v in inputs.items()}

        if isinstance(inputs, (list, tuple)):
            return [self.add_noise(v) for v in inputs]

        return inputs

    def _get_noise(self, tensor):
        """
        得到要添加的噪声值

        """
        if self.is_fuzzed:
            return False
        if not ops.is_floating_point(tensor) or ops.numel(tensor) == 0:
            return False

        pert_value = FreeBenchmarkConst.PERT_VALUE_DICT.get(tensor.dtype)
        if not pert_value:
            return False
        else:
            self.perturbation_value = pert_value

        max_val = ops.max(ops.abs(tensor))[0].item()
        if max_val < pert_value:
            return False

        noise = ops.full(tensor.shape, self.perturbation_value, dtype=tensor.dtype)
        return noise
