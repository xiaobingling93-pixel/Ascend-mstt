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

from msprobe.mindspore.common.const import Const, FreeBenchmarkConst
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.handler.handler_factory import HandlerFactory
from msprobe.mindspore.free_benchmark.perturbation.perturbation_factory import PerturbationFactory


class ForwardSelfChecker:

    def __init__(self, api_name: str):
        self.api_name = api_name

    def handle(self, params: HandlerParams):
        """
        装饰器实际执行逻辑

        """
        perturbation = PerturbationFactory.create(self.api_name)
        params.fuzzed_result = perturbation.handle(params)
        params.original_result = params.original_func(*params.args, **params.kwargs)
        if params.fuzzed_result is not False:
            return self.deal_fuzzed_and_original_result(params)
        return params.original_result

    def get_compare_data(self, params: HandlerParams):
        if self.api_name not in Const.COMMUNICATION_API_LIST:
            return
        # 以下为通讯类api处理逻辑
        params.fuzzed_result = params.fuzzed_value
        if Config.pert_type == FreeBenchmarkConst.IMPROVE_PRECISION:
            params.original_result = params.args
        else:
            params.original_result = params.args[params.index]

    def deal_fuzzed_and_original_result(self, params: HandlerParams):
        original_result = params.original_result
        self.get_compare_data(params)
        handler = HandlerFactory.create(self.api_name)
        result = handler.handle(params)
        if self.api_name in Const.COMMUNICATION_API_LIST:
            result = original_result
        return result
