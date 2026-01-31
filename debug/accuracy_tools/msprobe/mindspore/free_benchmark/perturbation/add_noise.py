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
            logger.warning(f"{self.api_name_with_id} can not add noise.")
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
