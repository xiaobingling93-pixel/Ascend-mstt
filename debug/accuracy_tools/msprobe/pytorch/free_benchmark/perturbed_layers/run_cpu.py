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
