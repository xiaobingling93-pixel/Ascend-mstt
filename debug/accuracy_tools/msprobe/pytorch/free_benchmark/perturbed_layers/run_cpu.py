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
from msprobe.pytorch.free_benchmark.common.params import DataParams
from msprobe.pytorch.free_benchmark.common.utils import Tools
from msprobe.pytorch.free_benchmark.common.enums import DeviceType
from msprobe.pytorch.free_benchmark.perturbed_layers.base_layer import BaseLayer


class CpuLayer(BaseLayer):

    def handle(self, params: DataParams):

        logger.info_on_rank_0(
            f"[msprobe] Free benchmark: Perturbation is to_cpu of {self.api_name}."
        )
        new_args = Tools.convert_device_and_dtype(params.args, DeviceType.CPU, change_dtype=True)
        new_kwargs = Tools.convert_device_and_dtype(params.kwargs, DeviceType.CPU, change_dtype=True)
        params.perturbed_result = params.origin_func(*new_args, **new_kwargs)
        return params.perturbed_result
