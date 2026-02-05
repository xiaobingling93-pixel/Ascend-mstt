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

import abc
import itertools
import logging
import sys
import time
import argparse
from datetime import datetime, timezone, timedelta
from multiprocessing import Pool
from typing import Iterator, List, Tuple

sys.path.append("./")
from tinker.profiler.profile_classes import ProfileArgs
from tinker.search.process import ResultOutputHandler
from tinker.search.data import SearchArgs, ResultArgs, Metrics, TaskParam, StageData
from tinker.search.cost_model import TinkerCostModel
from tinker.search.arguments import print_args, preprocess_args

from tinker.utils.utils import read_file, load_infos, convert_to_num_layers
from tinker.utils.logger import logger, init_log

MAX_FLOAT = 1.0e9
PRECISION_REDUNDANCY = 1.0e-7


class Optimizer(abc.ABC):

    def __init__(self, cost_model: TinkerCostModel, user_args, ):
        self.cost_model = cost_model
        self.user_args = user_args

    @abc.abstractmethod
    def search_parallel_strategies(self) -> List[Tuple[ResultArgs, Metrics]]:
        pass

    @abc.abstractmethod
    def process_result(self, strategy_metrics_pairs: List[Tuple[ResultArgs, Metrics]]):
        pass

    def optimize(self):
        result_pairs = self.search_parallel_strategies()
        self.process_result(result_pairs)


class TinkerOptimizer(Optimizer):

    def __init__(self, cost_model: TinkerCostModel, user_args):
        super().__init__(cost_model, user_args)
        self.script = self.read_pretrain_file()

    def search_parallel_strategies(self) -> List[Tuple[ResultArgs, Metrics]]:
        task_params = self._gen_task_params()
        strategy_metrics_list = self._parallel_task(task_params)
        flattened_list = list(itertools.chain(*strategy_metrics_list))
        return flattened_list

    def read_pretrain_file(self):
        # 若用户输入了 pretrain 的脚本路径，但该文件不存在，则报错
        if self.user_args.pretrain_script_path is not None:
            # 读文件
            logger.info('find pretrain script, will write top strategies into it')
            try:
                script = read_file(self.user_args.pretrain_script_path)
            except (FileNotFoundError, RuntimeError):
                logger.error(f'an error occurred when read file \'{self.user_args.pretrain_script_path}\'')
                raise
        else:
            script = ''
            logger.info('the pretrain script path is empty in user input, will write top strategies into a blank file')
        logger.info('result will store in %s', self.user_args.config_save_path)
        return script

    def process_result(self, result_pairs: List[Tuple[ResultArgs, Metrics]]):
        if not result_pairs:
            logger.info("no feasible config, exit")
            return

        result_output_handler = ResultOutputHandler(self.user_args, self.cost_model, result_pairs, self.script)

        # 对于result_pairs进行排序
        result_output_handler.sort()

        # 日志打屏以及写入sh文件, 默认取top 10, 完整结果存入csv
        result_output_handler.print_and_write_to_file(10)

    def _gen_task_params(self):
        # task 是什么定义？ task的目的是为了生成 num_intervals_list，共同组成最终的并行策略
        args = self.user_args
        cost_model = self.cost_model
        profiled_args_list = cost_model.get_profile_arg_list()  # type: List[ProfileArgs]
        # 3. 通过观测数据，搜索相关策略
        task_params = []
        # 这里的逻辑应该是除 num_layer_list 之外的所有参数
        for profiled_args in profiled_args_list:
            profiled_args: ProfileArgs
            # 计算当前profiled_args下的 pp dist_opt dp 取值范围
            num_npus = args.num_npus
            # TODO 该类约束统一处理
            if num_npus % profiled_args.tp:
                continue
            # stage变量的搜索空间生成
            pp_space = TinkerCostModel.get_pp_range(num_npus, args.num_layers, profiled_args)  # type: Iterator
            dist_opt_space = [0] if isinstance(profiled_args.ep, int) and profiled_args.ep > 1 else [0, 1]
            recompute_space = [0, 1]  # TODO 支持逐block重计算，当前使用统一full recompute

            # 生成任务队列
            for pp, dist_opt, recompute in itertools.product(pp_space, dist_opt_space, recompute_space):
                dp = num_npus // pp // profiled_args.tp
                local_batch_size = dp * profiled_args.mbs
                if args.global_batch_size % local_batch_size or dp == 1 and dist_opt:
                    continue
                search_args = SearchArgs(
                    pp=pp,
                    dp=dp,
                    recompute=recompute,
                    dist_opt=dist_opt,
                    **profiled_args.__dict__  # 继承 profiled_args 的所有属性
                )
                blocks = self.cost_model.init_blocks(profiled_args, self.user_args.num_layers)
                for block in blocks:
                    block.update_cost_model_args({
                        "dp": dp,
                        "dist_opt": search_args.dist_opt,
                        "recompute": search_args.recompute
                    })
                # 头尾处理不做recompute
                for block in [blocks[0], blocks[-2], blocks[-1]]:
                    block.recompute = False
                
                task_param = TaskParam(search_args=search_args, blocks=blocks)
                task_params.append(task_param)
        return task_params

    def _parallel_task(self, task_params: List[TaskParam]):
        # 寻找最优的几种划分方式
        if self.user_args.cpus <= 1:
            results = [self._memory_and_rounds_search(task_param) for task_param in task_params]
        else:
            with Pool(self.user_args.cpus) as pool:
                results = pool.map(self._memory_and_rounds_search, task_params)
        return results

    def _memory_and_rounds_search(self, task_param: TaskParam):
        search_round = 5
        # 用于存储一些 memory_limit 较小但 time_cost 稍大的组合
        best_results = []
        next_memory_limit = self.user_args.memory_limit
        # 计算保留内存
        reserved_mems = TinkerCostModel.calc_reserved_mem_costs(task_param.search_args.pp, task_param.blocks)
        # 动态计算memory_limits
        while search_round > 0:
            memory_limits = [next_memory_limit - reserved_mem for reserved_mem in reserved_mems]
            interval_layer_list = self._dynamic_programming(task_param, memory_limits)
            if not interval_layer_list:
                break

            num_layers = convert_to_num_layers(interval_layer_list)
            strategy = ResultArgs(
                gbs=self.user_args.global_batch_size,
                num_layers_list=num_layers,
                blocks=task_param.blocks,
                **task_param.search_args.__dict__
            )
            metrics = self.cost_model.calculate_cost(task_param, interval_layer_list)
            best_results.append((strategy, metrics))
            search_round -= 1
            # float 精度原因，输出的next_memory_limit可能会和输入相同，导致并行策略重复，此处减一微小值
            next_memory_limit = metrics.mem_cost - PRECISION_REDUNDANCY

        return best_results

    def _dynamic_programming(self, param: TaskParam, memory_limits: List[float]):
        """
        指定 memory_limit 下的最优结果
        @param param: 入参
        @param memory_limits: 各stages的reserved内存开销，刻画内存碎片
        @return: 最优结果
        """
        num_all_blocks = len(param.blocks)
        profile_args = param.blocks[0].profile_args
        micro_batch_num = self.user_args.global_batch_size // param.search_args.dp // profile_args.mbs
        pp = param.search_args.pp

        # 头尾处理不流水线切分约束
        head_min_num = 1
        end_min_num = 2
        # dp[i][j] i：block_num，j: stage_idx
        dp = [[StageData(num_npu_before=0, stage_time_max_min=float('inf'), num_layer_list=list(), stage_mem_max=0)]
              * (pp + 1) for _ in range(num_all_blocks + 1)]
        # 动规方程定义：前i个block划分为j个stage的所有方式中，最大time_cost的最小值
        dp[0][0] = StageData(num_npu_before=0, stage_time_max_min=0, num_layer_list=list(), stage_mem_max=0)

        for j in range(1, pp + 1):

            for i in range(1, num_all_blocks + 1):
                if i <= head_min_num:
                    # 约束一
                    continue

                for k in range(i - 1, -1, -1):
                    current_blocks = param.blocks[k: i]
                    # 约束二：
                    if j == param.search_args.pp and len(current_blocks) <= end_min_num:
                        continue

                    # 使用j-1，提前固定乘数
                    num_fwd_act = TinkerCostModel.get_num_fwd_act(pp, j - 1, micro_batch_num)
                    current_stage_mem = TinkerCostModel.get_stage_mem_cost(current_blocks, num_fwd_act)
                    # 使用stage对应内存上限判断当前是否可以提前退出
                    if current_stage_mem >= memory_limits[j - 1]:
                        # 倒序，可以break
                        break
                    # 计算第j个stage的时间
                    current_max_status = dp[k][j - 1]
                    num_npu_before, time_cost, _, _ = self.cost_model.get_stage_status(
                        current_blocks, current_max_status.num_npu_before, j == 1, j == pp
                    )
                    # 当前最佳的切分方式
                    current_max_time_cost = max(dp[k][j - 1].stage_time_max_min, time_cost)
                    current_max_mem_cost = max(dp[k][j - 1].stage_mem_max, current_stage_mem)
                    if current_max_time_cost < dp[i][j].stage_time_max_min:
                        idx_list = dp[k][j - 1].num_layer_list
                        current_list = idx_list.copy()
                        current_list.append(k)
                        dp[i][j] = StageData(num_npu_before=num_npu_before, stage_time_max_min=current_max_time_cost,
                                             num_layer_list=current_list, stage_mem_max=current_max_mem_cost)

        best_result = dp[num_all_blocks][pp]
        if not best_result.num_layer_list:
            return None
        # 根据分割点，计算划分区间
        points = best_result.num_layer_list
        points.append(num_all_blocks)
        dynamic_stage_intervals = list()
        for i in range(pp):
            start_idx = points[i]
            end_idx = points[i + 1]
            dynamic_stage_intervals.append((start_idx, end_idx - 1))
        return dynamic_stage_intervals


def initialize(args):
    init_log(None, logging.INFO)
    load_infos(args)
    preprocess_args(args)
    # 准备logger
    formatted_time = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d-%H-%M-%S')
    init_log(args.log_file, log_level=logging.INFO)
    logger.info(
        f"[LOG][SEARCH]({formatted_time}) start searching for {args.model_name}, {args.model_size}, {args.num_nodes}"
        f" nodes * {args.num_npus_per_node} NPUs.")
    print_args(args)


def run(args: argparse.Namespace):
    if args.mode != 'all' and args.mode != 'search':
        return
    start_time = time.time()
    initialize(args)
    # 1. 实例化CostModel
    cost_model = TinkerCostModel(args)
    optimizer = TinkerOptimizer(cost_model=cost_model, user_args=args)
    optimizer.optimize()
    end_time = time.time()
    logger.info(f"[TOTAL TIME] {end_time - start_time} s.")
