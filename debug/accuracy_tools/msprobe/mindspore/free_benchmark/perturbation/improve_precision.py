from typing import Any

import mindspore as ms
from mindspore import Tensor, ops

from msprobe.mindspore.free_benchmark.perturbation.base_perturbation import BasePerturbation
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.common.const import Const


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
