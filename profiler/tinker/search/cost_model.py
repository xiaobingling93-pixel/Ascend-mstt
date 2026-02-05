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

import csv
import glob
import logging
import math
import os
import sys
import time
import argparse
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple, Union

sys.path.append("./")

from tinker.profiler.profile_classes import ProfileArgs
from tinker.search.arguments import print_args, preprocess_args
from tinker.search.data import TaskParam, SearchArgs, ResultArgs, Metrics
from tinker.search.process import ResultOutputHandler
from tinker.utils.block_args import BlockArgs, BlockCost, DetailedInfo
from tinker.utils.utils import load_infos, convert_to_pp_stage_block_idx
from tinker.utils.logger import logger, init_log

FeaturesType = ProfileArgs
ProfileDataType = Dict[FeaturesType, Dict[str, float]]


class FixedValueDict:
    def __init__(self, fixed_value):
        self.fixed_value = fixed_value

    def __getitem__(self, key):
        return self.fixed_value

    def get(self, key, default=None):
        return self.fixed_value


class ProfiledData:
    def __init__(self):
        # 现在主要就用这玩意儿
        self._block_data = defaultdict(dict)  # type: Dict[FeaturesType, Dict[str, BlockCost]]

    @staticmethod
    def _get_data(datas: ProfileDataType, features: FeaturesType, block_name="") -> Union[Dict, FixedValueDict, float]:
        if features not in datas:
            logger.info(f"feature {features} not in profiled data, using 10000000.0")
            if block_name:
                return 20000000.0
            return FixedValueDict(20000000.0)
        if block_name:
            return datas[features][block_name]
        return datas[features]

    def add_data(self, data: Tuple[float, ...], features: FeaturesType, block_name: str):
        self._block_data[features][block_name] = BlockCost(*data)


    def get_data_by_args(self, profiled_args: ProfileArgs):
        return self._block_data[profiled_args]

    def get_profiled_args_list(self) -> List[FeaturesType]:
        return list(self._block_data.keys())

    def get_block_names(self):
        """返回"""
        for block_data in self._block_data.values():
            if isinstance(block_data, dict):
                return list(block_data.keys())

        return []


@dataclass
class BlockNames:
    """需要与block_profiler打配合，有点不好"""
    pre: str
    block: str
    post1: str
    post2: str


class TinkerCostModel:
    def __init__(self, args):
        self._band_data_ready = None
        self._block_data_ready = None
        self.profiled_data = ProfiledData()
        # TODO 考虑把读数据的模块单独提一个类，或者相关read逻辑放进 ProfiledData
        self._read_block_data(args.profiled_data_path)
        self.inter_band = None  # type: Optional[List[float]]
        self.intra_band = None  # type: Optional[List[float]]
        self._read_band_time(args.profiled_data_path)
        self.block_names = BlockNames(*self.profiled_data.get_block_names())
        # 剔除transformer block块后的其余块个数
        self.num_other_block = len(fields(self.block_names)) - 1
        self.num_procs_per_node = args.num_npus_per_node
        self.args = args

    @property
    def _data_ready(self):
        return self._band_data_ready and self._block_data_ready

    @staticmethod
    def calc_reserved_mem_costs(pp: int, blocks: List[BlockArgs]) -> List[float]:
        """
        为每个stage内存加上额外的reserved部分(内存碎片导致)，当前策略为
        1. 含头处理的stage: blocks[0].bwd_reserved
        2. 含尾处理的stage: 尾处理峰值工作内存，`blocks[-1].data.act + blocks[-1].fwd_reserved`
        3. 其他stage: blocks[1].bwd_reserved * 2
        4. 若含头又含尾，则: max(blocks[0].bwd_reserved, blocks[-1].fwd_reserved)
        """
        reserved_mem_costs = []
        first_stage_mem_reserved = blocks[0].max_reserved_mem
        last_stage_mem_reserved = blocks[-1].max_reserved_mem
        other_stage_mem_reserved = blocks[1].max_reserved_mem
        if pp == 1:
            reserved_mem_costs.append(max(first_stage_mem_reserved, last_stage_mem_reserved, other_stage_mem_reserved))
            return reserved_mem_costs

        reserved_mem_costs.append(max(first_stage_mem_reserved, other_stage_mem_reserved))
        for _ in range(1, pp - 1):
            reserved_mem_costs.append(other_stage_mem_reserved)
        reserved_mem_costs.append(max(last_stage_mem_reserved, other_stage_mem_reserved))
        return reserved_mem_costs

    @staticmethod
    def get_num_fwd_act(pp: int, stage: int, micro_batch_num: int) -> int:
        """
        给出指定stage做1F1B调度时需保存的峰值前向激活值份数

        :param pp: 本次训练流水线并行度，也即总流水线stage数量
        :param stage: 当前stage序号，首stage序号为0
        :param micro_batch_num: 在流水线上的微批个数，即gbs // dp // mbs
        :return: 该stage需保存你的峰值前向激活值份数
        """
        return min(pp - stage, micro_batch_num)

    @staticmethod
    def get_stage_mem_cost(current_blocks, num_fwd_act):
        """
        计算stage的内存开销
        """
        mem_cost = 0
        head_block = current_blocks[0]
        head_block.is_first = True
        for block in current_blocks:
            block.num_fwd_act = num_fwd_act
            mem_cost += block.block_mem()

        # 重计算开销
        mem_cost += TinkerCostModel.calc_recompute_mem(current_blocks)
        head_block.is_first = False
        return mem_cost

    @staticmethod
    def calc_recompute_mem(blocks: List[BlockArgs]):
        recompute_work_block = max(blocks, key=lambda x: x.data.act if x.recompute else 0)
        recompute_work_mem = recompute_work_block.data.act if recompute_work_block.recompute else 0

        return recompute_work_mem

    @staticmethod
    def get_pp_range(num_npus, num_layers, p_args: ProfileArgs):
        for pp in range(1, min(num_layers, num_npus) + 1):
            if num_npus % (p_args.npu_used * pp) == 0 and num_npus // p_args.tp // pp >= p_args.ep:
                yield pp

    @staticmethod
    def _read_band_file(file_path: str):
        with open(file_path) as f:
            src_data = csv.reader(f)
            _ = next(src_data)
            row = next(src_data)
            return [float(band) for band in row]

    @staticmethod
    def _refresh_blocks(param):
        for block in param.blocks:
            block.num_fwd_act = None
            block.is_first = False

    def node_comm_time(self, data_size, inter_node=True):
        """返回用于计算p2p通信时间的通信时间 inter_node用于指定是否节点间通信 向下取整到2的幂次"""
        if not self._band_data_ready:
            raise RuntimeError("band data not ready yet, run `_read_band_time` first.")
        if data_size < 0:
            raise ValueError(f'communicate data size invalid: {data_size} <= 0')
        if data_size == 0:
            # TODO 这个情况，得区分：是不存在通信，还是有一个空通信。前者取0，后者取传包最小时间
            return 0
        bands = self.inter_band if inter_node else self.intra_band
        index = int(math.log(data_size, 2))
        if index >= 1:
            index -= 1
        if index >= len(bands):
            band = bands[-1] * 0.001
        else:
            band = bands[index] * 0.001
        return data_size / band

    def p2p_comm_time(self, block_args: BlockArgs, num_npu_before: int, head=False, tail=False):
        if not head and not tail:
            raise ValueError("When calculate p2p communicate time, either head or tail should be set to True")
        comm_size = block_args.data.in_size if head else block_args.data.out_size
        is_cross_nodes = num_npu_before % self.num_procs_per_node == 0 and num_npu_before
        comm_time = self.node_comm_time(comm_size, is_cross_nodes)
        return comm_time

    def get_profile_arg_list(self) -> List[ProfileArgs]:
        return self.profiled_data.get_profiled_args_list()

    def get_block_args(self, block_name: str, profiled_args: ProfileArgs) -> BlockArgs:
        data = self.profiled_data.get_data_by_args(profiled_args)
        if block_name not in data:
            raise KeyError(f"{block_name} is not defined in profiled_data")
        block_data = data[block_name]
        return BlockArgs(self.args, profiled_args, block_data)

    def init_blocks(self, profile_args: ProfileArgs, num_layers: int) -> List[BlockArgs]:
        """当前就是头处理 + 若干个block + 尾处理，调用时机确定 ProfileArgs 之后"""
        # 头处理块
        block_list = [self.get_block_args(self.block_names.pre, profile_args)]  # type: List[BlockArgs]
        # transformer block
        block_list.extend([self.get_block_args(self.block_names.block, profile_args) for _ in range(num_layers)])
        # 尾处理块
        block_list.append(self.get_block_args(self.block_names.post1, profile_args))
        block_list.append(self.get_block_args(self.block_names.post2, profile_args))
        # transformer block 注入仅使用一次的 attention_mask 尺寸
        attention_mask_mem = self.args.seq_length * self.args.seq_length / 1024.0 / 1024.0
        for block in block_list[1:-2]:
            block.attention_mask_mem = attention_mask_mem
        return block_list

    def get_stage_status(self, current_blocks, num_npu_before, is_first_stage, is_last_stage):
        """
        此处计算与stage有关的time_cost
        """
        time_cost = 0
        head_block = current_blocks[0]
        tail_block = current_blocks[-1]
        for block in current_blocks:
            time_cost += block.block_time()

        # 头尾通信开销
        input_comm = 0 if is_first_stage else self.p2p_comm_time(head_block, num_npu_before, head=True)
        num_npu_before += head_block.num_npu_block
        output_comm = 0 if is_last_stage else self.p2p_comm_time(tail_block, num_npu_before, tail=True)
        time_cost += input_comm + output_comm
        return num_npu_before, time_cost, input_comm, output_comm

    def calculate_cost(self, param: TaskParam, pp_stage_block_intervals: list, detail=False):
        if detail:
            detail_infos = []
        time_costs = []
        mem_costs = []
        num_npu_before = 0
        profile_args = param.blocks[0].profile_args
        micro_batch_num = self.args.global_batch_size // param.search_args.dp // profile_args.mbs
        # 提前计算各stage因内存碎片而产生的reserved内存峰值
        pp = param.search_args.pp
        reserved_mem_costs = TinkerCostModel.calc_reserved_mem_costs(pp, param.blocks)
        # 逐 pp stage 计算时空开销
        for p in range(pp):
            if detail:
                logger.info(f'stage {p}'.center(80, '='))
                detailed_info = DetailedInfo()
                detail_infos.append(detailed_info)
            time_cost, mem_cost = 0, 0
            head_idx, tail_idx = pp_stage_block_intervals[p]
            head_block, tail_block = param.blocks[head_idx], param.blocks[tail_idx]
            # 首block属性更改
            head_block.is_first = True
            # 逐block计算性能
            for block_idx in range(head_idx, tail_idx + 1):
                block = param.blocks[block_idx]
                block.num_fwd_act = TinkerCostModel.get_num_fwd_act(pp, p, micro_batch_num)
                mem_cost += block.block_mem()
            num_npu_before, time_cost, input_comm, output_comm = self.get_stage_status(
                param.blocks[head_idx: tail_idx + 1], num_npu_before, p == 0, p == pp - 1)

            # stage 重计算内存、内存碎片
            recompute_mem = TinkerCostModel.calc_recompute_mem(param.blocks[head_idx:tail_idx + 1])
            mem_cost += recompute_mem
            mem_cost += reserved_mem_costs[p]
            time_costs.append(time_cost)
            mem_costs.append(mem_cost)
            if detail:
                for block_idx in range(head_idx, tail_idx + 1):
                    block = param.blocks[block_idx]
                    _ = block.block_time(detail=detail, detail_info=detailed_info)
                    _ = block.block_mem(detail=detail, detail_info=detailed_info)
                detailed_info.set_and_print(input_comm, output_comm, recompute_mem, reserved_mem_costs[p], mem_cost)
                logger.info('stage %d total Memory: %.3f MB', p, mem_cost)

        bubble_time = sum(time_costs)
        profile_args = param.blocks[0].profile_args
        micro_batch_num = self.args.global_batch_size // param.search_args.dp // profile_args.mbs
        time_costs = [bubble_time + (micro_batch_num - 1) * stage_time for stage_time in time_costs]
        if detail:
            # 输出时间仿真器细分数据
            logger.info(f'Time Cost with Bubble'.center(80, '='))
            logger.info('Sum(unit time): %.3f ms', bubble_time / 1000)
            for time_cost, detail_info in zip(time_costs, detail_infos):
                detail_info.print_time(bubble_time, micro_batch_num, time_cost)
        # 参数还原，避免后续影响
        self._refresh_blocks(param)
        return Metrics(time_costs, mem_costs, max(time_costs), max(mem_costs))

    def _read_block_data(self, data_path: str):
        """基于profiler，生成searcher参数范围；或者直接基于每个tp sp mbs [ep]，去衍化dp pp zero"""
        file_path = os.path.join(data_path, 'profiled_data.csv')
        try:
            with open(file_path, 'r') as f:
                data = csv.reader(f)
                next(data, None)
                for row in data:
                    if all(not field.strip() for field in row):
                        continue
                    filename_without_suffix = row[0]
                    block_name = row[1]
                    data = tuple(float(data) for data in row[2:])
                    profile_args = ProfileArgs.new_from_file_name(filename_without_suffix)
                    self.profiled_data.add_data(data, profile_args, block_name)
        except Exception as e:
            raise RuntimeError(f'Load profiled data: {file_path} failed.') from e
        self._block_data_ready = True

    def _read_band_time(self, data_path):
        # TODO 当前p2p.csv的表头在读取数据时无用，且囿于2的幂次，考虑优化
        intra_band_file = os.path.join(data_path, "p2p_intra_node.csv")
        inter_band_file = os.path.join(data_path, "p2p_inter_node.csv")
        logger.info(intra_band_file)
        logger.info(inter_band_file)

        # 读取`节点内`带宽数据
        try:
            self.intra_band = self._read_band_file(intra_band_file)
        except FileNotFoundError:
            logger.error(f"intra-node bandwidth file is not found.")

        # 读取`节点间`带宽数据
        try:
            self.inter_band = self._read_band_file(inter_band_file)
        except FileNotFoundError:
            logger.error(f"inter-node bandwidth file is not found, using intra-node bandwidth instead.")
            self.inter_band = self.intra_band

        if self.inter_band is None and self.intra_band is None:
            raise RuntimeError("Intra bandwidth and intra bandwidth file are required.")
        self._band_data_ready = True


def run(args: argparse.Namespace):
    if args.mode != 'simulate':
        return
    # 这个入口设置成debug
    init_log(None, logging.DEBUG)
    start_time = time.time()
    preprocess_args(args)
    load_infos(args)
    print_args(args)
    # 1. 实例化CostModel
    cost_model = TinkerCostModel(args)
    # 2. 从观测数据中用户指定的仿真所需数据 args -> pred_profiled_args
    pred_profiled_args = ProfileArgs(tp=args.simu_tp, sp=args.simu_sp, ep=args.simu_ep, mbs=args.micro_batch_size)
    # 3. 计算开销
    # 3.1 生成子图
    pred_blocks = cost_model.init_blocks(pred_profiled_args, args.num_layers)
    # 3.2 校验所给策略有效性
    remainder = args.num_npus % (args.simu_pp * pred_profiled_args.tp)
    if remainder != 0:
        raise ValueError(
            "incorrect num_npus={}, pp={}, tp={}, the former must be divided into the latter two.".format(
                args.num_npus, args.simu_pp, pred_profiled_args.tp
            ))
    # 3.3 计算DP LBS，打包CostModel变量并刷新block
    npu_used = pred_profiled_args.tp * args.simu_pp
    if args.num_npus % npu_used:
        raise ValueError("num_npus cannot be evenly divided by the parallel strategy, check tp pp")
    dp = args.num_npus // npu_used
    local_batch_size = dp * pred_profiled_args.mbs
    if args.global_batch_size % local_batch_size:
        raise ValueError("incorrect gbs={}, dp={}, mbs={}, the former must be divided into the latter two.".format(
            args.global_batch_size, dp, args.micro_batch_size
        ))
    cost_model_args = dict(dp=dp, dist_opt=args.dist_opt, recompute=args.recompute)
    # 当前所有block统一cost_model_args，尤其是recompute
    for block in pred_blocks:
        block.update_cost_model_args(cost_model_args)
    # 头尾处理不做recompute
    for block in [pred_blocks[0], pred_blocks[-2], pred_blocks[-1]]:
        block.recompute = False

    # 3.4 转换pp切分num_layer_list
    split_way = list(map(int, args.num_layer_list.split(',')))
    intervals = convert_to_pp_stage_block_idx(split_way, len(pred_blocks))
    # 3.5 计算开销，传入detail开关
    search_args = SearchArgs(pp=args.simu_pp, **cost_model_args, **pred_profiled_args.__dict__)
    task_param = TaskParam(search_args=search_args, blocks=pred_blocks)
    strategy = ResultArgs(
        gbs=args.global_batch_size,
        num_layers_list=args.num_layer_list,
        blocks=task_param.blocks,
        **task_param.search_args.__dict__
    )
    metrics = cost_model.calculate_cost(task_param, intervals, args.detail)
    result_output_handler = ResultOutputHandler(args, cost_model, [(strategy, metrics)])
    result_output_handler.print_and_write_to_file(1, save=False)

    end_time = time.time()
    logger.info(f"[TOTAL TIME] {end_time - start_time} s.")
