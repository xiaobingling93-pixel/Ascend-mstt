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


import torch
from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.enums import PerturbationMode
from msprobe.pytorch.free_benchmark.common.params import DataParams
from msprobe.pytorch.free_benchmark.perturbed_layers.npu.npu_base_layser import (
    NpuBaseLayer,
)


class NoChangeLayer(NpuBaseLayer):

    def no_change(self, tensor_obj):
        """
        不对输入做任何改变、直接二次执行
        """
        self.is_added = True
        return tensor_obj

    def handle(self, params: DataParams):
        """
        对输入添加扰动并返回
        """
        logger.info_on_rank_0(
            f"[msprobe] Free benchmark: Perturbation is "
            f"{PerturbationMode.NO_CHANGE} of {self.api_name}."
        )
        params.perturbed_value = self.no_change(params.args[params.valid_input_index])
        return self.perturbed_result(params)
