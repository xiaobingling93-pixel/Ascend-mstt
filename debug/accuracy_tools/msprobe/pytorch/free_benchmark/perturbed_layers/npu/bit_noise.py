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
from msprobe.pytorch.free_benchmark.common.constant import ThresholdConfig
from msprobe.pytorch.free_benchmark.common.enums import PerturbationMode
from msprobe.pytorch.free_benchmark.common.params import DataParams
from msprobe.pytorch.free_benchmark.common.utils import TorchC
from msprobe.pytorch.free_benchmark.perturbed_layers.npu.npu_base_layser import (
    NpuBaseLayer,
)


class BitNoiseLayer(NpuBaseLayer):
    def __init__(self, api_name):
        super().__init__(api_name)
        self.bit_mode = TorchC.bitwise_xor
        self.bit_tail: int = 1
        self.bit_type = None

    def add_bit_noise(self, tensor_obj):
        """
        对输入添加噪声
        """
        # finfo应该列入黑名单

        if isinstance(tensor_obj, torch.Tensor):
            self._set_perturbation_bit(tensor_obj)
            if not self.pre_check(tensor_obj):
                return tensor_obj
            sub_normal = torch.finfo(tensor_obj.dtype).smallest_normal
            noise = TorchC.full(
                tensor_obj.shape,
                self.bit_tail,
                device=tensor_obj.device,
                dtype=self.bit_type,
            )
            result = tensor_obj.view(self.bit_type)
            result = TorchC.where(
                TorchC.gt(TorchC.abs(tensor_obj), sub_normal),
                self.bit_mode(result, noise),
                result,
            ).view(tensor_obj.dtype)

            self.is_added = True
            return result
        if isinstance(tensor_obj, dict):
            return {key: self.add_bit_noise(value) for key, value in tensor_obj.items()}
        if isinstance(tensor_obj, (tuple, list)):
            return type(tensor_obj)([self.add_bit_noise(value) for value in tensor_obj])
        return tensor_obj

    def handle(self, params: DataParams):
        """
        对输入添加扰动并返回
        """
        logger.info_on_rank_0(
            f"[msprobe] Free benchmark: Perturbation is "
            f"{PerturbationMode.BIT_NOISE} of {self.api_name}."
        )
        params.perturbed_value = self.add_bit_noise(params.args[params.valid_input_index])
        return self.perturbed_result(params)

    def _check_details(self, tensor_obj):
        """
        判断是否需要添加扰动,  bit翻转
        """
        if not self.bit_type:
            logger.info_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.api_name}, "
                f"dtype unsupported. Cancel perturbation."
            )
            return False
        if tensor_obj.numel() == 0:
            logger.warning_on_rank_0(
                f"[msprobe] Free benchmark: For {self.api_name}, tensor shape must > 0"
                f" Cancel adding noise."
            )
            return False
        abs_tol = ThresholdConfig.ABS_TOL_VALUE_DICT.get(
            tensor_obj.dtype, ThresholdConfig.NOISE_INPUT_LOWER_BOUND
        )
        try:
            max_val = TorchC.max(TorchC.abs(tensor_obj)).item()
        except Exception:
            logger.warning_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.api_name}, "
                f"when calculate maximun value, tensor is changed to float32."
            )
            max_val = TorchC.max(TorchC.abs(tensor_obj.to(torch.float32))).item()
        if max_val < abs_tol:
            logger.info_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.api_name}, "
                f"Maximun value is less than the  minimun threshold. Cancel add noise."
            )
            return False
        return True

    def _set_perturbation_bit(self, tensor_obj):
        """
        根据不同浮点数确定不同位数扰动值
        """
        bit_len_type = ThresholdConfig.PERTURBATION_BIT_DICT.get(tensor_obj.dtype)
        if bit_len_type:
            self.bit_tail = 1
            self.bit_type = bit_len_type
