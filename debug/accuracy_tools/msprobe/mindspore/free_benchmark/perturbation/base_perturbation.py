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


from typing import Any

from msprobe.core.common.const import Const
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.common.utils import Tools


class BasePerturbation:

    def __init__(self, api_name_with_id: str):
        self.api_name_with_id = api_name_with_id
        self.is_fuzzed = False
        self.perturbation_value = None

    @staticmethod
    def get_fuzzed_result(params: HandlerParams):
        if Config.stage == Const.BACKWARD:
            fuzzed_result = Tools.get_grad(params.original_func, *params.args[:params.index],
                                           params.fuzzed_value, *params.args[params.index + 1:], **params.kwargs)

            if fuzzed_result is None:
                return False
        else:
            fuzzed_result = params.original_func(*params.args[:params.index], params.fuzzed_value,
                                                 *params.args[params.index + 1:], **params.kwargs)
        return fuzzed_result

    def handler(self, params: HandlerParams) -> Any:
        pass
