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

import mindspore as ms
from mindspore import Tensor, ops

from msprobe.mindspore.common.const import Const
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.perturbation.base_perturbation import BasePerturbation


class ImprovePrecisionPerturbation(BasePerturbation):

    def improve_tensor_precision(self, target_tensor):
        if isinstance(target_tensor, Tensor) and ops.is_floating_point(target_tensor) and \
           target_tensor.dtype not in [ms.float64, ms.float32]:
            self.is_fuzzed = True
            return target_tensor.to(ms.float32)
        if isinstance(target_tensor, dict):
            return {k: self.improve_tensor_precision(v) for k, v in target_tensor.items()}
        if isinstance(target_tensor, (tuple, list)):
            return type(target_tensor)([self.improve_tensor_precision(v) for v in target_tensor])
        return target_tensor

    def handle(self, params: HandlerParams) -> Any:
        args = self.improve_tensor_precision(params.args)
        kwargs = self.improve_tensor_precision(params.kwargs)
        fuzzed_value = args
        if self.api_name in Const.COMMUNICATION_API_LIST:
            params.fuzzed_value = fuzzed_value
        if not self.is_fuzzed:
            logger.warning(f"{self.api_name} can not improve precision.")
            return False
        return params.original_func(*args, **kwargs)
