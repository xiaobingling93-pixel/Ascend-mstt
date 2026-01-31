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
