# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from mindspore import Tensor

from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams


class FixHandler:

    def __init__(self, api_name_with_id: str):
        self.api_name_with_id = api_name_with_id

    @staticmethod
    def use_fuzzed_result(original_result, fuzzed_result):
        if isinstance(original_result, Tensor):
            return fuzzed_result.to(original_result.dtype)
        if isinstance(original_result, dict):
            dict_fixed_result = dict()
            for k, v in original_result.items():
                dict_fixed_result[k] = FixHandler.use_fuzzed_result(v, fuzzed_result[k])
            return dict_fixed_result
        if isinstance(original_result, (tuple, list)):
            list_fixed_result = list()
            for i, v in enumerate(original_result):
                list_fixed_result.append(FixHandler.use_fuzzed_result(v, fuzzed_result[i]))
            return type(original_result)(list_fixed_result)
        return original_result

    def handle(self, params: HandlerParams) -> Any:
        try:
            return FixHandler.use_fuzzed_result(params.original_result, params.fuzzed_result)
        except Exception as e:
            logger.error(f"{self.api_name_with_id} failed to fix.")
            logger.error(str(e))
            return params.original_result
