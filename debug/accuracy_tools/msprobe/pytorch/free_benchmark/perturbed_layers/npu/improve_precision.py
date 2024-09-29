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
from msprobe.core.common.const import Const
from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.constant import CommonField
from msprobe.pytorch.free_benchmark.common.enums import PerturbationMode
from msprobe.pytorch.free_benchmark.common.params import DataParams
from msprobe.pytorch.free_benchmark.perturbed_layers.npu.npu_base_layser import (
    NpuBaseLayer,
)


class ImprovePrecisionLayer(NpuBaseLayer):

    def improve_tensor_precision(self, tensor_obj):
        if (
            isinstance(tensor_obj, torch.Tensor)
            and torch.is_floating_point(tensor_obj)
            and tensor_obj.dtype not in [torch.float32, torch.float64]
        ):
            self._set_improve_values(tensor_obj)
            tensor_obj = self._change_dtype(tensor_obj)
            self.is_added = True
            return tensor_obj
        if isinstance(tensor_obj, dict):
            return {
                key: self.improve_tensor_precision(value)
                for key, value in tensor_obj.items()
            }
        if isinstance(tensor_obj, (tuple, list)):
            return type(tensor_obj)(
                [self.improve_tensor_precision(value) for value in tensor_obj]
            )
        return tensor_obj

    def handle(self, params: DataParams):
        logger.info_on_rank_0(
            f"[msprobe] Free benchmark: Perturbation is "
            f"{PerturbationMode.IMPROVE_PRECISION} of {self.api_name}."
        )
        new_args = self.improve_tensor_precision(params.args)
        if params.fuzz_stage == Const.BACKWARD:
            new_kwargs = {}
        else:
            new_kwargs = self.improve_tensor_precision(params.kwargs)
        # 如果输入中全为高精度、应跳过二次执行、减少多余显存引用
        if not self.is_added:
            return params.perturbed_result
        if "inplace" in new_kwargs:
            new_kwargs["inplace"] = False
        params.perturbed_result = params.origin_func(*new_args, **new_kwargs)
        return params.perturbed_result

    def _set_improve_values(self, inputs):
        if inputs.dtype in [torch.float16, torch.bfloat16]:
            self.perturbed_value = torch.float32

    def _change_dtype(self, inputs):
        if hasattr(inputs, CommonField.DEVICE):
            device = inputs.device
            if device is CommonField.META:
                new_inputs = inputs.to(
                    device=CommonField.META, dtype=self.perturbed_value
                )
            else:
                new_inputs = inputs.to(dtype=self.perturbed_value).to(device)
        else:
            new_inputs = inputs.to(dtype=self.perturbed_value)
        return new_inputs
