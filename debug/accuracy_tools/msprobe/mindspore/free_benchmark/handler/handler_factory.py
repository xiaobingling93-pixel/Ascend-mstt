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

from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.handler.check_handler import CheckHandler
from msprobe.mindspore.free_benchmark.handler.fix_handler import FixHandler


class HandlerFactory:
    result_handlers = {
        FreeBenchmarkConst.CHECK: CheckHandler,
        FreeBenchmarkConst.FIX: FixHandler
    }

    @staticmethod
    def create(api_name_with_id: str):
        handler = HandlerFactory.result_handlers.get(Config.handler_type)
        if handler:
            return handler(api_name_with_id)
        else:
            logger.error(f"{Config.handler_type} is not supported.")
            raise Exception
