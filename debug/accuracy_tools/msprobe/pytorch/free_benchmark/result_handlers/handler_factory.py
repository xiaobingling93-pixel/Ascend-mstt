# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


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
