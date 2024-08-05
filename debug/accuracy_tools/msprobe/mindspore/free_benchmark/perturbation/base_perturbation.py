from typing import Any

from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams


class BasePerturbation:

    def __init__(self, api_name: str):
        self.api_name = api_name
        self.is_fuzzed = False
        self.perturbation_value = None

    @staticmethod
    def get_fuzzed_result(params: HandlerParams):
        args_front = params.args[:params.index]
        args_rear = params.args[params.index + 1:]
        fuzzed_result = params.original_func(*args_front, params.fuzzed_value, *args_rear, **params.kwargs)
        return fuzzed_result

    def handler(self, params: HandlerParams) -> Any:
        pass
