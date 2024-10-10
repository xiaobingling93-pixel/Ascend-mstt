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
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import mindspore as ms
from mindspore import Tensor, ops

from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.common.utils import Tools


class BaseHandler(ABC):

    def __init__(self, api_name: str):
        self.api_name = api_name

    @staticmethod
    def pre_calculate(original_output, fuzzed_output):
        abs_tol = FreeBenchmarkConst.PERT_VALUE_DICT.get(fuzzed_output.dtype,
                                                         FreeBenchmarkConst.PERT_VALUE_DICT.get(ms.float32))

        return original_output.to(fuzzed_output.dtype), fuzzed_output, abs_tol

    @staticmethod
    def get_threshold(dtype):
        err = Tools.get_default_error_threshold(dtype)
        return err

    @staticmethod
    def convert_overflow_ratio_to_consistent(ratio):
        if math.isnan(ratio) or math.isinf(ratio):
            return FreeBenchmarkConst.NO_CHANGE_ERROR_THRESHOLD
        return ratio

    @staticmethod
    def get_endless_norm(first_tensor, second_tensor, abs_tol):
        if first_tensor.dtype != ms.bfloat16 and second_tensor.dtype != ms.bfloat16:
            ratio_tensor1 = ops.where(ops.abs(second_tensor) > abs_tol, ops.div(first_tensor, second_tensor), 1)
            ratio_tensor2 = ops.where(ops.abs(first_tensor) > abs_tol, ops.div(second_tensor, first_tensor), 1)
        else:
            ratio_tensor1 = ops.where(ops.abs(second_tensor).to(ms.float32) > abs_tol,
                                      ops.div(first_tensor.to(ms.float32), second_tensor.to(ms.float32)), 1)
            ratio_tensor2 = ops.where(ops.abs(first_tensor).to(ms.float32) > abs_tol,
                                      ops.div(second_tensor.to(ms.float32), first_tensor.to(ms.float32)), 1)
        norm1 = BaseHandler.convert_overflow_ratio_to_consistent(ops.max(ratio_tensor1)[0].to(ms.float32).item())
        norm2 = BaseHandler.convert_overflow_ratio_to_consistent(ops.max(ratio_tensor2)[0].to(ms.float32).item())
        norm3 = BaseHandler.convert_overflow_ratio_to_consistent(ops.min(ratio_tensor1)[0].to(ms.float32).item())
        ratio = FreeBenchmarkConst.SYMBOL_FLIPPING_RATIO if norm3 < 0 else max(norm1, norm2)

        return ratio

    @staticmethod
    def ratio_calculate(original_output, fuzzed_output) -> float:
        try:
            original_output, fuzzed_output, abs_tol = BaseHandler.pre_calculate(original_output, fuzzed_output)
        except Exception as e:
            logger.error(f"When computing ratio, y1 or y2 dtype is not supported {str(e)}")
            return FreeBenchmarkConst.NO_CHANGE_ERROR_THRESHOLD

        abs_tol = abs_tol ** 0.5

        return BaseHandler.get_endless_norm(original_output, fuzzed_output, abs_tol)

    @staticmethod
    def npu_compare(original_output, fuzzed_output) -> Tuple[bool, Optional[float]]:
        if not isinstance(fuzzed_output, Tensor):
            logger.error(f"The compare for output type `{type(fuzzed_output)}` is not supported")
            return True, 1.0

        # 范数计算等
        err_thd = BaseHandler.get_threshold(original_output.dtype)
        ratio = BaseHandler.ratio_calculate(original_output, fuzzed_output)
        is_consistent = err_thd >= ratio >= 1.0 / err_thd
        return is_consistent, ratio

    @staticmethod
    def is_float_tensor(output) -> bool:
        if isinstance(output, Tensor) and ops.is_floating_point(output):
            return True
        if isinstance(output, (list, tuple)):
            for i in output:
                if isinstance(i, Tensor) and ops.is_floating_point(i):
                    return True
        return False

    @abstractmethod
    def handle(self, params: HandlerParams) -> Any:
        pass
