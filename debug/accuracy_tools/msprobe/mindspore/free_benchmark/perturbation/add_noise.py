from typing import Any

from mindspore import Tensor, ops

from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.perturbation.base_perturbation import BasePerturbation
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.common.const import FreeBenchmarkConst


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

    def _get_noise(self, input):
        """
        得到要添加的噪声值

        """
        if self.is_fuzzed:
            return False
        if not ops.is_floating_point(input) or ops.numel(input) == 0:
            return False

        pert_value = FreeBenchmarkConst.PERT_VALUE_DICT.get(input.dtype)
        if not pert_value:
            return False
        else:
            self.perturbation_value = pert_value

        max_val = ops.max(ops.abs(input))[0].item()
        if max_val < pert_value:
            return False

        noise = ops.full(input.shape, self.perturbation_value, dtype=input.dtype)
        return noise
