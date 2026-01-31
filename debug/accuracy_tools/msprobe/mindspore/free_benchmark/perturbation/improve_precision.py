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

import mindspore as ms
from mindspore import Tensor, ops

from msprobe.core.common.const import Const
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.common.utils import Tools
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
        if not self.is_fuzzed:
            logger.warning(f"{self.api_name_with_id} can not improve precision.")
            return False

        if Config.stage == Const.BACKWARD:
            fuzzed_result = Tools.get_grad(params.original_func, *args, **kwargs)
            if fuzzed_result is not None:
                return fuzzed_result
            else:
                return False

        return params.original_func(*args, **kwargs)
