# -*- coding: utf-8 -*-
# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

""" search 过程中的数据类
"""
from dataclasses import dataclass, fields
from typing import List, Optional

from tinker.utils.block_args import BlockArgs
from tinker.profiler.profile_classes import ProfileArgs


@dataclass(frozen=True)
class SearchArgs(ProfileArgs):
    """
    搜索参数，继承自ProfileArgs

    属性:
        mbs: int - micro batch size，微批次大小
        tp: int - tensor parallel size
        sp: int - sequence parallel size
        ep: int - expert parallel
        dp: Optional[int] - dp值，表示数据并行的副本数
        pp: Optional[int] - pp值，表示pipeline并行的阶段数
        recompute: Optional[int] - 重计算
        dist_opt: Optional[int] - 分布式优化器
    """
    dp: Optional[int] = None  # dp值，表示数据并行的副本数
    pp: Optional[int] = None  # pp值，表示pipeline并行的阶段数
    recompute: Optional[int] = None  # 重计算
    dist_opt: Optional[int] = None  # 分布式优化器

    def __post_init__(self):
        '''检查'recompute'和'dist_opt'这两个属性的值是否在{0, 1}中'''
        valid_values = {0, 1}
        for f in filter(lambda f: f.name in {'recompute', 'dist_opt'}, fields(self)):
            if f.name in {'recompute', 'dist_opt'} and getattr(self, f.name) not in valid_values:
                raise ValueError(f"Invalid value for {f.name}, expected 0 or 1, got {getattr(self, f.name)}")


@dataclass(frozen=True)
class ResultArgs(SearchArgs):
    """
    搜索结果参数，继承自SearchArgs

    属性:
        mbs: int - micro batch size，微批次大小
        tp: int - tensor parallel size
        sp: int - sequence parallel size
        ep: int - expert parallel
        dp: Optional[int] - dp值，表示数据并行的副本数
        pp: Optional[int] - pp值，表示pipeline并行的阶段数
        recompute: Optional[int] - 重计算
        dist_opt: Optional[int] - 分布式优化器
        num_layers_list: str - 表示不同阶段的神经网络层数配置列表
    """
    num_layers_list: Optional[str] = None  # 示例: 4,4,4,4，网络层数配置列表，每个元素对应不同阶段的层数
    gbs: Optional[int] = None  # 全局的批次大小，表示一次处理的数据量
    blocks: Optional[List[BlockArgs]] = None


@dataclass()
class Metrics:
    """
    性能数据
    """
    time_costs: list
    mem_costs: list
    time_cost: float
    mem_cost: float
    tokens_per_npu_per_sec: Optional[float] = None


@dataclass(frozen=True)
class TaskParam:
    """
    任务维度的参数，作为非均匀区间划分时的传参
    """
    search_args: SearchArgs
    blocks: List[BlockArgs]


@dataclass(frozen=True)
class StageData:
    """
    用于存储动规过程中的数据
    """
    num_npu_before: int
    stage_time_max_min: float
    num_layer_list: list
    stage_mem_max: float
