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
