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

from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.perturbation.base_perturbation import BasePerturbation


class ExchangeValuePerturbation(BasePerturbation):

    @staticmethod
    def _check_tensor_shape(inputs):
        dims = len(inputs.shape)
        if dims == 1 and inputs.shape[0] > 1:
            return True
        if dims > 1 and inputs.shape[1] > 0:
            if inputs.shape[0] > 1 or inputs.shape[1] > 1:
                return True
        return False

    def handle(self, params: HandlerParams) -> Any:
        """
        返回首尾交换后的api输出

        """
        params.fuzzed_value = self.exchange_value(params.args[params.index])
        if not self.is_fuzzed:
            logger.warning(f"{self.api_name_with_id} can not exchange value.")
            return False
        return self.get_fuzzed_result(params)

    def exchange_value(self, inputs) -> Any:
        """
        返回首尾交换后的api输入

        """
        if isinstance(inputs, Tensor) and ops.is_floating_point(inputs):
            if self.is_fuzzed or not self._check_tensor_shape(inputs):
                return inputs
            result = inputs.copy()
            if len(inputs.shape) == 1:
                first_element = inputs[0]
                last_element = inputs[-1]
                result[0] = last_element
                result[-1] = first_element
            else:
                first_element = inputs[0][0]
                last_element = inputs[-1][-1]
                result[0][0] = last_element
                result[-1][-1] = first_element

            self.is_fuzzed = True
            return result

        if isinstance(inputs, dict):
            return {k: self.exchange_value(v) for k, v in inputs.items()}

        if isinstance(inputs, (list, tuple)):
            return type(inputs)([self.exchange_value(v) for v in inputs])

        return inputs
