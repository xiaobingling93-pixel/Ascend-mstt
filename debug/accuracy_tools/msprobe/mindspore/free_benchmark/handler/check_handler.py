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


from dataclasses import asdict
from typing import Any

from mindspore import Tensor, ops

from msprobe.core.data_dump.json_writer import DataWriter
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.common.utils import make_unequal_row
from msprobe.mindspore.free_benchmark.handler.base_handler import BaseHandler


class CheckHandler(BaseHandler):

    def npu_compare_and_save(self, original_output, fuzzed_output, params: HandlerParams, output_index=None):
        is_consistent, ratio = self.npu_compare(original_output, fuzzed_output)
        params.is_consistent = params.is_consistent and is_consistent
        if not is_consistent:
            row = make_unequal_row(self.api_name_with_id, params, ratio, output_index)
            data_dict = asdict(row)
            DataWriter.write_data_to_csv(
                data_dict.values(),
                data_dict.keys(),
                Config.dump_path
            )
            logger.error(f"{self.api_name_with_id} is not consistent")

    def handle(self, params: HandlerParams) -> Any:
        try:
            if not self.is_float_tensor(params.fuzzed_result):
                return
            if isinstance(params.fuzzed_result, Tensor):
                self.npu_compare_and_save(params.original_result, params.fuzzed_result, params)
            elif isinstance(params.fuzzed_result, (list, tuple)):
                for i, item in enumerate(params.original_result):
                    if ops.is_tensor(item) and ops.is_floating_point(item):
                        self.npu_compare_and_save(item, params.fuzzed_result[i], params, output_index=i)
        except Exception as e:
            logger.error(str(e))
