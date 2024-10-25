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

from dataclasses import dataclass
from typing import Any, Optional

import mindspore as ms
from mindspore import Tensor

from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.runtime import Runtime


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
    fuzzed_tensor = params.fuzzed_result
    if index is not None:
        original_tensor = original_tensor[index]
        fuzzed_tensor = fuzzed_tensor[index]
        row.output_index = index
    if isinstance(original_tensor, Tensor):
        row.dtype = original_tensor.dtype
        row.shape = original_tensor.shape
    row.rank = Runtime.rank_id if Runtime.rank_id != -1 else None
    return row
