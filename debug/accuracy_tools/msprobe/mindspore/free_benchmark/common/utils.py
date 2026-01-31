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


from dataclasses import dataclass
from typing import Any, Optional

import mindspore as ms
from mindspore import Tensor, ops

from msprobe.core.common.runtime import Runtime
from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams


class Tools:

    @staticmethod
    def get_first_tensor_dtype(tensor_seq: Any):
        if isinstance(tensor_seq, Tensor):
            return tensor_seq.dtype
        if isinstance(tensor_seq, (list, tuple)):
            for i in tensor_seq:
                if isinstance(i, Tensor):
                    return i.dtype
        raise Exception("The sequence does not contain tensors.")

    @staticmethod
    def get_default_error_threshold(dtype):
        if Config.pert_type == FreeBenchmarkConst.NO_CHANGE:
            return FreeBenchmarkConst.NO_CHANGE_ERROR_THRESHOLD
        return FreeBenchmarkConst.ERROR_THRESHOLD.get(dtype, FreeBenchmarkConst.ERROR_THRESHOLD.get(ms.float32))

    @staticmethod
    def get_grad_out(outputs):
        if isinstance(outputs, Tensor):
            return ops.ones_like(outputs)
        if isinstance(outputs, (tuple, list)):
            return type(outputs)([Tools.get_grad_out(v) for v in outputs])
        return outputs

    @staticmethod
    def get_grad(func, *args, **kwargs):
        def target_func(*inputs):
            return func(*inputs, **kwargs)

        outputs, vjp_fn = ms.vjp(target_func, *args)
        values = Tools.get_grad_out(outputs)
        return vjp_fn(values)


@dataclass
class UnequalRow:
    rank: Optional[int] = None
    pert_type: Optional[str] = None
    stage: Optional[str] = None
    step: Optional[int] = None
    api_name: Optional[str] = None
    max_rel: Optional[float] = None
    dtype: Optional[str] = None
    shape: Optional[str] = None
    output_index: Optional[int] = None


def make_unequal_row(
    api_name: str,
    params: HandlerParams,
    ratio: float = None,
    index: int = None,
):
    row = UnequalRow(
        api_name=api_name,
        pert_type=Config.pert_type,
        output_index=index,
        stage=Config.stage,
        step=Runtime.step_count
    )
    if isinstance(ratio, float):
        row.max_rel = ratio - 1
    original_tensor = params.original_result
    if index is not None:
        original_tensor = original_tensor[index]
        row.output_index = index
    if isinstance(original_tensor, Tensor):
        row.dtype = original_tensor.dtype
        row.shape = original_tensor.shape
    row.rank = Runtime.rank_id if Runtime.rank_id != -1 else None
    return row
