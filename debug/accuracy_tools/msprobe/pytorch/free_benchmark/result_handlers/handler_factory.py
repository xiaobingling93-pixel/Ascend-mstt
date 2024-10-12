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

from msprobe.pytorch.free_benchmark import FreeBenchmarkException
from msprobe.pytorch.free_benchmark.common.constant import PreheatConfig
from msprobe.pytorch.free_benchmark.common.enums import HandlerType
from msprobe.pytorch.free_benchmark.common.params import HandlerParams
from msprobe.pytorch.free_benchmark.result_handlers.check_handler import CheckerHandler
from msprobe.pytorch.free_benchmark.result_handlers.preheat_handler import PreheatHandler
from msprobe.pytorch.free_benchmark.result_handlers.fix_handler import FixHandler


class FuzzHandlerFactory:

    result_handlers = {
        HandlerType.CHECK: CheckerHandler,
        HandlerType.FIX: FixHandler,
        HandlerType.PREHEAT: PreheatHandler,
    }

    @staticmethod
    def create(params: HandlerParams):
        if_preheat = params.preheat_config.get(PreheatConfig.IF_PREHEAT)
        if not if_preheat:
            handler = FuzzHandlerFactory.result_handlers.get(params.handler_type)
        else:
            handler = FuzzHandlerFactory.result_handlers.get(HandlerType.PREHEAT)
        if not handler:
            raise FreeBenchmarkException(
                FreeBenchmarkException.UnsupportedType,
                f"无标杆工具支持 [ {HandlerType.CHECK}、{HandlerType.FIX}] 形式",
            )
        return handler(params)
