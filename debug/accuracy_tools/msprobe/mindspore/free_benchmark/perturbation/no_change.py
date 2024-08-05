from typing import Any

from msprobe.mindspore.free_benchmark.perturbation.base_perturbation import BasePerturbation
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams


class NoChangePerturbation(BasePerturbation):

    def handle(self, params: HandlerParams) -> Any:
        params.fuzzed_value = params.args[params.index]
        self.is_fuzzed = True
        return self.get_fuzzed_result(params)
