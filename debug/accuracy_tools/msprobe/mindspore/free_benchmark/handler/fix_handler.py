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
