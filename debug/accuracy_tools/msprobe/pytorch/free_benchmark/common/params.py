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
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from msprobe.core.common.exceptions import FreeBenchmarkException
from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.enums import (
    DeviceType,
    FuzzLevel,
    PerturbationMode,
)
from msprobe.pytorch.free_benchmark.common.utils import Tools


@dataclass
class DataParams:
    args: Optional[Tuple] = None
    kwargs: Optional[Dict] = None
    valid_input_index: Optional[int] = None
    original_result: Optional[Any] = None
    perturbed_result: Optional[Any] = None
    is_consistent: Optional[bool] = True
    perturbed_value: Optional[Any] = None
    origin_func: Optional[Callable] = None
    api_type: Optional[str] = None
    fuzz_stage: Optional[str] = None


@dataclass
class HandlerParams:
    handler_type: Optional[str] = None
    api_name: Optional[str] = None
    pert_mode: Optional[PerturbationMode] = None
    step: Optional[int] = None
    fuzz_stage: Optional[str] = None
    fuzz_device: Optional[DeviceType] = None
    preheat_config: Optional[Dict] = None
    fuzz_level: Optional[str] = None


@dataclass
class UnequalRow:
    rank: Optional[int] = None
    pert_mode: Optional[PerturbationMode] = None
    stage: Optional[str] = None
    step: Optional[int] = None
    api_name: Optional[str] = None
    max_rel: Optional[float] = None
    dtype: Optional[str] = None
    shape: Optional[str] = None
    output_index: Optional[int] = None


@dataclass
class BenchmarkThd:
    rtol: Optional[float] = None  # 相对误差阈值
    small_value: Optional[float] = None  # 小值域
    small_value_atol: Optional[float] = None  # 小值域绝对阈值
    err_balance: Optional[float] = None  # 误差均衡性


def check_args_type(args: Tuple) -> int:
    for i, arg in enumerate(args):
        if torch.is_tensor(arg):
            if arg.is_meta:
                continue
            if not torch.is_floating_point(arg):
                continue
            return i
        if isinstance(arg, (List, Tuple, Dict)):
            return i
    return -1


def data_pre_deal(name, func, args, kwargs):
    data_params = DataParams(args=args, kwargs=kwargs, origin_func=func)
    index = check_args_type(args)
    data_params.valid_input_index = index
    if index == -1:
        logger.warning_on_rank_0(
            f"[msprobe] Free benchmark: 无标杆工具不支持当前算子的输入类型 {name}."
        )
    return data_params


def make_handler_params(name, config, step):
    handler_params = HandlerParams()
    handler_params.api_name = name
    handler_params.step = step
    handler_params.handler_type = config.handler_type
    handler_params.fuzz_stage = config.fuzz_stage
    handler_params.fuzz_device = config.fuzz_device
    handler_params.preheat_config = config.preheat_config
    handler_params.fuzz_level = config.fuzz_level
    handler_params.pert_mode = config.pert_mode
    return handler_params


def make_unequal_row(
    data_params: DataParams,
    handle_params: HandlerParams,
    ratio: float = None,
    index: int = None,
):
    row = UnequalRow(
        api_name=handle_params.api_name,
        pert_mode=handle_params.pert_mode,
        output_index=index,
        stage=handle_params.fuzz_stage,
        step=handle_params.step,
    )
    if isinstance(ratio, float):
        row.max_rel = ratio - 1
    if isinstance(ratio, str):
        row.max_rel = ratio
    origin_tensor = data_params.original_result
    perturbed_tensor = data_params.perturbed_result
    if index is not None:
        if index >= len(origin_tensor) or index >= len(perturbed_tensor):
            err_msg = f"When generating unequal results, index {index} of output is out of bounds. please check!"
            raise FreeBenchmarkException(
                FreeBenchmarkException.OutputIndexError,
                error_info=err_msg,
            )
        origin_tensor = origin_tensor[index]
        perturbed_tensor = perturbed_tensor[index]
        row.output_index = index
    if isinstance(origin_tensor, torch.Tensor):
        row.dtype = origin_tensor.dtype
        row.shape = origin_tensor.shape
    row.rank = Tools.get_dist_rank()
    # 以下暂不支持
    if handle_params.fuzz_level == FuzzLevel.ADV_LEVEL:
        pass
    if handle_params.fuzz_level == FuzzLevel.REAL_LEVEL:
        pass
    return row
