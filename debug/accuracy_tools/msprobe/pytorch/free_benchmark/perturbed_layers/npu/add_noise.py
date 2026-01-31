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
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.constant import ThresholdConfig
from msprobe.pytorch.free_benchmark.common.enums import PerturbationMode
from msprobe.pytorch.free_benchmark.common.params import DataParams
from msprobe.pytorch.free_benchmark.common.utils import TorchC
from msprobe.pytorch.free_benchmark.perturbed_layers.npu.npu_base_layser import (
    NpuBaseLayer,
)


class AddNoiseLayer(NpuBaseLayer):

    @recursion_depth_decorator("FreeBenchmark: AddNoiseLayer.add_noise")
    def add_noise(self, tensor_obj):
        if isinstance(tensor_obj, torch.Tensor):
            self.perturbed_value = ThresholdConfig.PERTURBATION_VALUE_DICT.get(
                tensor_obj.dtype
            )
            if not self.pre_check(tensor_obj):
                return tensor_obj
            noise = self._get_noise(tensor_obj)
            result = TorchC.where(
                TorchC.gt(TorchC.abs(tensor_obj), self.perturbed_value ** 0.5),
                TorchC.add(noise, tensor_obj),
                tensor_obj,
            ).to(tensor_obj.dtype)
            self.is_added = True
            return result
        if isinstance(tensor_obj, dict):
            return {key: self.add_noise(value) for key, value in tensor_obj.items()}
        if isinstance(tensor_obj, (tuple, list)):
            return type(tensor_obj)([self.add_noise(value) for value in tensor_obj])
        return tensor_obj

    def handle(self, params: DataParams):
        """
        对输入添加扰动并返回
        """
        logger.info_on_rank_0(
            f"[msprobe] Free benchmark: Perturbation is "
            f"{PerturbationMode.ADD_NOISE} of {self.api_name}."
        )
        params.perturbed_value = self.add_noise(params.args[params.valid_input_index])
        return self.perturbed_result(params)

    def _get_noise(self, tensor_obj):
        dtype = tensor_obj.dtype
        device = str(tensor_obj.device)
        noise = TorchC.full(
            tensor_obj.shape,
            self.perturbed_value,
            device=device,
            dtype=dtype,
        )
        return noise

    def _check_details(self, tensor_obj):
        """
        判断是否需要添加扰动
        """
        if not self.perturbed_value:
            logger.warning_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.api_name}, "
                f"dtype unsupported. Cancel perturbation."
            )
            return False
        if tensor_obj.numel() == 0:
            logger.warning_on_rank_0(
                f"[msprobe] Free benchmark: For {self.api_name}, tensor shape must > 0."
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
                f"when calculating the maximum value, the tensor is changed to float32."
            )
            max_val = TorchC.max(TorchC.abs(tensor_obj.to(torch.float32))).item()
        if max_val < abs_tol:
            logger.warning_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.api_name}, "
                f"maximum value is less than the minimum threshold. Cancel adding noise."
            )
            return False
        return True
