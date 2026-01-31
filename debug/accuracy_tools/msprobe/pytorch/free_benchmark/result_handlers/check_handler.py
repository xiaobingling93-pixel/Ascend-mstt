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

from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.enums import DeviceType
from msprobe.pytorch.free_benchmark.common.params import DataParams, make_unequal_row
from msprobe.pytorch.free_benchmark.common.utils import Tools
from msprobe.pytorch.free_benchmark.compare.single_benchmark import SingleCompare
from msprobe.pytorch.free_benchmark.result_handlers.base_handler import FuzzHandler


class CheckerHandler(FuzzHandler):
    def other_compare(self, data_params: DataParams) -> bool:
        is_consistent = SingleCompare().compare_seq(
                    data_params.original_result, data_params.perturbed_result
                )
        if not is_consistent:
            self.unequal_rows.append(
                make_unequal_row(data_params, self.params)
            )

    def get_threshold(self, dtype):
        return self._get_default_threshold(dtype)

    def handle(self, data_params: DataParams) -> Any:
        if isinstance(data_params.perturbed_result, bool) or not Tools.is_float_tensor(
            data_params.perturbed_result
        ):
            return data_params.original_result
        try:
            if self.params.fuzz_device == DeviceType.NPU:
                self.cmp_output_npu(data_params)
            else:
                self.other_compare(data_params)
        except Exception as e:
            logger.warning_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.params.api_name}, "
                f"when comparing the results, an exception is raised: {e}"
            )
        return data_params.original_result
