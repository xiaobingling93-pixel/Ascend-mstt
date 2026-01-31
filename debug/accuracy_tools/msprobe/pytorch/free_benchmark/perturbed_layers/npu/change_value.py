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
from msprobe.pytorch.free_benchmark.common.enums import PerturbationMode
from msprobe.pytorch.free_benchmark.common.params import DataParams
from msprobe.pytorch.free_benchmark.common.utils import TorchC
from msprobe.pytorch.free_benchmark.perturbed_layers.npu.npu_base_layser import (
    NpuBaseLayer,
)


class ChangeValueLayer(NpuBaseLayer):
    def __init__(self, api_name):
        super().__init__(api_name)
        self.head: int = 0
        self.tail: int = -1

    @recursion_depth_decorator("FreeBenchmark: ChangeValueLayer.change_value")
    def change_value(self, tensor_obj):
        """
        交换张量首尾
        """
        if isinstance(tensor_obj, torch.Tensor) and self.pre_check(tensor_obj):
            new_tensor = TorchC.clone(tensor_obj)
            if new_tensor.ndim == 1:
                temp_first = TorchC.clone(new_tensor[self.head])
                temp_last = TorchC.clone(new_tensor[self.tail])
                new_tensor[self.head] = temp_last
                new_tensor[self.tail] = temp_first
            else:
                temp_first = TorchC.clone(new_tensor[self.head][self.head])
                temp_last = TorchC.clone(new_tensor[self.tail][self.tail])
                new_tensor[self.head][self.head] = temp_last
                new_tensor[self.tail][self.tail] = temp_first

            self.is_added = True
            return new_tensor
        if isinstance(tensor_obj, dict):
            return {key: self.change_value(value) for key, value in tensor_obj.items()}
        if isinstance(tensor_obj, (tuple, list)):
            return type(tensor_obj)([self.change_value(value) for value in tensor_obj])
        return tensor_obj

    def handle(self, params: DataParams):
        """
        对输入添加扰动并返回
        """
        logger.info_on_rank_0(
            f"[msprobe] Free benchmark: Perturbation is "
            f"{PerturbationMode.CHANGE_VALUE} of {self.api_name}."
        )
        params.perturbed_value = self.change_value(params.args[params.valid_input_index])
        return self.perturbed_result(params)

    def _check_details(self, tensor_obj):
        """
        判断是否需要添加扰动,  首尾值交换
        """
        # 对于维度大于1的张量、要求1维至少大于1且0维和1维至少一个长度大于2
        if tensor_obj.ndim > 1:
            if tensor_obj.size(1) == 0 or (tensor_obj.size(1) < 2 and tensor_obj.size(0) < 2):
                logger.info_on_rank_0(
                    f"[msprobe] Free Benchmark: For {self.api_name} with ndim {tensor_obj.ndim}, "
                    f"at least one of 0-dimension or 1-dimension greater than 1. Cancel change value."
                )
                return False
        # 不支持维度等于0的张量、对于维度等于1的张量、要求0维长度大于2
        elif tensor_obj.dim() == 0 or tensor_obj.size(0) < 2:
            logger.info_on_rank_0(
                f"[msprobe] Free Benchmark: For {self.api_name}, "
                f"0-dimension must greater than 1. Cancel change value."
            )
            return False
        return True
