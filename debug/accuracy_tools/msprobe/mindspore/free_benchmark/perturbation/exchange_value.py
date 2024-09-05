from typing import Any

from mindspore import Tensor

from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.perturbation.base_perturbation import BasePerturbation
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams


class ExchangeValuePerturbation(BasePerturbation):

    def handle(self, params: HandlerParams) -> Any:
        """
        返回首尾交换后的api输出

        """
        params.fuzzed_value = self.exchange_value(params.args[params.index])
        if not self.is_fuzzed:
            logger.warning(f"{self.api_name} can not exchange value.")
            return False
        return self.get_fuzzed_result(params)

    def exchange_value(self, inputs) -> Any:
        """
        返回首尾交换后的api输入

        """
        if isinstance(inputs, Tensor):
            if not self.is_fuzzed and len(inputs.shape) > 0 and inputs.shape[0] > 1:
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
