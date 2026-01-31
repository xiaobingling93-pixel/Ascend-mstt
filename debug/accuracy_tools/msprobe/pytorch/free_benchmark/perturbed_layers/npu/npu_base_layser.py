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


from abc import abstractmethod
from typing import Any

import torch
from msprobe.pytorch.free_benchmark.common.params import DataParams
from msprobe.pytorch.free_benchmark.perturbed_layers.base_layer import BaseLayer


class NpuBaseLayer(BaseLayer):
    def __init__(self, api_name: str) -> None:
        super().__init__(api_name)
        self.perturbed_value = None  # 扰动的元素
        self.is_added = False  # 标记当前算子输入是否调整

    @staticmethod
    def perturbed_result(params: DataParams) -> Any:
        args_front = params.args[: params.valid_input_index]
        args_rear = params.args[params.valid_input_index + 1:]
        # 此处会将有inplace属性的算子换为非inplace
        if "inplace" in params.kwargs:
            params.kwargs["inplace"] = False
        params.perturbed_result = params.origin_func(
            *args_front, params.perturbed_value, *args_rear, **params.kwargs
        )
        return params.perturbed_result

    @abstractmethod
    def handle(self, params: DataParams) -> Any:
        pass

    def pre_check(self, tensor_obj):
        """
        检查张量是否符合标准(float类型且最大值大于对应精度最小值)
        """
        # 只针对第一个满足要求的添加扰动
        if self.is_added:
            return False
        if not torch.is_floating_point(tensor_obj):
            return False
        if not self._check_details(tensor_obj):
            return False
        return True

    def _check_details(self, tensor_obj):
        return True
