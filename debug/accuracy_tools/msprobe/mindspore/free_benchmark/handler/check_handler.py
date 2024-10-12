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
            row = make_unequal_row(self.api_name, params, ratio, output_index)
            data_dict = asdict(row)
            DataWriter.write_data_to_csv(
                data_dict.values(),
                data_dict.keys(),
                Config.dump_path
            )
            logger.error(f"{self.api_name} is not consistent")

    def handle(self, params: HandlerParams) -> Any:
        try:
            if not self.is_float_tensor(params.fuzzed_result):
                return params.original_result
            if isinstance(params.fuzzed_result, Tensor):
                self.npu_compare_and_save(params.original_result, params.fuzzed_result, params)
            elif isinstance(params.fuzzed_result, (list, tuple)):
                for i, item in enumerate(params.original_result):
                    if ops.is_tensor(item) and ops.is_floating_point(item):
                        self.npu_compare_and_save(item, params.fuzzed_result[i], params, output_index=i)
        except Exception as e:
            logger.error(str(e))
        return params.original_result
