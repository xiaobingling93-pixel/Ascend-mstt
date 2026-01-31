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

from msprobe.core.common.exceptions import FreeBenchmarkException
from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.params import DataParams
from msprobe.pytorch.free_benchmark.common.utils import Tools
from msprobe.pytorch.free_benchmark.result_handlers.base_handler import FuzzHandler


class FixHandler(FuzzHandler):

    def get_threshold(self, dtype):
        return self._get_default_threshold(dtype)

    def handle(self, data_params: DataParams) -> Any:
        try:
            return Tools.convert_fuzz_output_to_origin(
                data_params.original_result, data_params.perturbed_result
            )
        except FreeBenchmarkException as e:
            logger.warning(
                f"[msprobe] Free Benchmark: For {self.params.api_name} "
                f"Fix output failed because of: \n{e}"
            )
            return data_params.original_result
