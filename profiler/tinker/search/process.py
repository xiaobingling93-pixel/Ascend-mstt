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

import os
from typing import List, Tuple, Dict
from dataclasses import asdict, fields

from tinker.search.data import ResultArgs, Metrics, TaskParam, SearchArgs
from tinker.utils.utils import extract_between, del_line, write_lines, convert_to_pp_stage_block_idx
from tinker.utils.logger import logger


class ResultOutputHandler:
    """
    ResultOutputHandler类，用于处理输出结果。
    """

    def __init__(self, args, cost_model, result_pairs: List[Tuple[ResultArgs, Metrics]], script=None):
        self.user_args = args
        self.cost_model = cost_model
        self.script = script
        self.result_pairs = result_pairs
        self.result_pairs_sorted = None
        self._calculate_tokens()


    @staticmethod
    def _write_strategy_to_file(script: str, strategy_param):
        """
        替换用户脚本中的指定参数，若为空，则返回
        :param script:
        :param strategy_param:
        :return:
        """

        # 直接建两个分类
        params_need_to_deleted = [
            '--tensor-model-parallel-size ', '--micro-batch-size ', '--global-batch-size ',
            '--sequence-parallel ', '--use-distributed-optimizer ', '--recompute-method ',
            '--recompute-granularity ', '--recompute-num-layers ', '--pipeline-model-parallel-size ',
            '--num-layer-list', '--context-parallel-size ', '--context-parallel-algo ',
            '--ulysses-degree-in-cp ', '--cp-attention-mask-type ', '--use-cp-send-recv-overlap ',
            '--kv-head-repeat-before-uly-alltoall ', '--num-layers-per-virtual-pipeline-stage ',
            '--overlap-grad-reduce ', '--overlap-param-gather ']

        params_need_to_append = [f'--tensor-model-parallel-size {strategy_param.tp} \\',
                                f'--micro-batch-size {strategy_param.mbs} \\',
                                f'--global-batch-size {strategy_param.gbs} \\',
                                f'--overlap-grad-reduce \\',
                                f'--pipeline-model-parallel-size {strategy_param.pp} \\']

        if strategy_param.pp > 1:
            params_need_to_append.append(f'--num-layer-list {strategy_param.num_layers_list} \\')

        if strategy_param.sp:
            params_need_to_append.append('--sequence-parallel \\')

        if strategy_param.dist_opt:
            params_need_to_append.append('--use-distributed-optimizer \\')

        if strategy_param.recompute:
            params_need_to_append.append('--recompute-method uniform \\')
            params_need_to_append.append('--recompute-granularity full \\')
            params_need_to_append.append('--recompute-num-layers 1 \\')

        # 插入首部：
        format_params = ['    ' + value for value in params_need_to_append]

        tinker_strategy_params = '\n'.join(format_params)
        tinker_search_args_str = 'TINKER_SEARCH_ARGS'
        tinker_strategy_params = f"{tinker_search_args_str}=\"\n{tinker_strategy_params}\n\""

        # 直接删完
        res = del_line(params_need_to_deleted, script)

        # 往首行插入 TINKER_SEARCH_ARGS
        res = '\n'.join([tinker_strategy_params, res])

        # 找到 torchrun xxx .py 或 python xxx.py，以便插入tinker参数
        run_key_words = ['torchrun', 'python']
        hit_key_word = None

        # 先找一圈，找不到直接报错
        for run_key_word in run_key_words:
            cmd_content = extract_between(run_key_word, 'py', res)
            if cmd_content is not None:
                hit_key_word = run_key_word
                break

        if hit_key_word is None:
            # 可能是空白文件或者没以上 run_key_words 的脚本，直接返回
            return res.splitlines()

        num_skip_line = len(cmd_content.splitlines())
        hit_key_word_idx = -1
        res_lines = res.splitlines()
        for idx, line in enumerate(res_lines):
            if hit_key_word in line:
                hit_key_word_idx = idx
                break

        # 在行尾命令行处，加上tinker 的参数
        insert_idx = hit_key_word_idx + num_skip_line
        tinker_args_in_cmd = ''.join(['      ${', tinker_search_args_str, '} \\'])
        res_lines.insert(insert_idx, tinker_args_in_cmd)
        return res_lines


    @staticmethod
    def _get_result_dict(rank: int, result_pair: Tuple[ResultArgs, Metrics]):
        strategy, metric = result_pair
        # 提取 info_text 中的指标值到 value_dict
        info_values = {
            "token_per_npu_per_sec": round(metric.tokens_per_npu_per_sec, 1),
            "time_cost": round(metric.time_cost / 1000, 3),
            "mem_cost": round(metric.mem_cost, 2)
        }

        # 构造完整的 value_dict
        value_dict = {
            **{
                "rank": rank + 1,
                "tp": strategy.tp,
                "pp": strategy.pp,
                "dp": strategy.dp,
                "sp": strategy.sp,
                "ep": strategy.ep,
                "dist_opt": strategy.dist_opt,
                "mbs": strategy.mbs,
                "num_layer_list": list(map(int, strategy.num_layers_list.split(','))),
                "recompute": strategy.recompute,
            },
            **info_values  # 合并 info_text 中的指标
        }

        return value_dict


    def sort(self):
        '''排序
        对于result_pairs，按照tokens_per_npu_per_sec的值从大到小排序，
        如果tokens_per_npu_per_sec的值相同，则根据time_cost的值从小到大排序，
        如果time_cost的值也相同，则根据mem_cost的值从小到大排序
        '''
        self.result_pairs_sorted = sorted(
            self.result_pairs,
            key=lambda item: (
                -item[1].tokens_per_npu_per_sec,
                item[1].time_cost,
                item[1].mem_cost
            )
        )


    def print_and_write_to_file(self, top_num, save=True):
        """
        将结果写入文件并打印日志表格
        :param top_num: 需要打印和写入文件的最优配置数量
        :param save: 是否将结果写入文件
        :return: 无
        :raise ValueError: 如果结果对的有序列表为空，抛出异常
        """
        # 如果不需要保存结果，将结果对的有序列表设置为结果对列表
        if not save:
            self.result_pairs_sorted = self.result_pairs
        # 如果结果对的有序列表为空，抛出异常
        if not self.result_pairs_sorted:
            raise ValueError('Please sort the result first!')

        # 初始化日志表格宽度字典
        table_widths = {}
        # 遍历结果对的有序列表，写入csv文件，存入top <top_num>到sh文件，更新日志表格宽度用于后续打印top <top_num>的配置
        for config_rank, result_pair in enumerate(self.result_pairs_sorted):
            # 如果需要保存结果，将结果对写入csv文件
            if save:
                self._write_to_csv(result_pair)
            # 获取结果对的字典形式
            value_dict = self._get_result_dict(config_rank, result_pair)
            if config_rank == 0:
                # 初始化日志表格宽度, 以表头的宽度为初始值
                table_widths = {v: len(str(v)) for v in value_dict.keys()}
            if config_rank + 1 <= top_num:
                # 只存 top 10的pretrain脚本
                if save and config_rank + 1 <= top_num:
                    self._write_to_sh(result_pair, config_rank + 1, self.script)
                # 更新表格宽度
                for k, v in value_dict.items():
                    temp_width = table_widths.get(k)
                    table_widths[k] = max(temp_width, len(str(v)))

        # 打印日志表格，呈现top top_num配置
        if save:
            self._print_table(table_widths, self.result_pairs_sorted[:top_num], f"top {top_num} configs", True)
        else:
            self._print_table(table_widths, self.result_pairs_sorted, "simulate config", False)

        # simulate模式下无需打印最优配置
        if save:
            # 打印最优配置
            best_strategy = self.result_pairs_sorted[0]
            self._print_table(table_widths, [best_strategy], "Best config", False)


    def _calculate_tokens(self):
        for result_pair in self.result_pairs:
            _, metric = result_pair
            tokens_per_npu_per_sec = (
                self.user_args.seq_length * self.user_args.global_batch_size /
                self.user_args.num_npus /
                metric.time_cost * 1000000
            )
            metric.tokens_per_npu_per_sec = tokens_per_npu_per_sec


    def _write_to_csv(self, result_pair: Tuple[ResultArgs, Metrics]):
        """
        将某result_pair写入csv文件

        参数:
        result_pair: 一个元组，包含ResultArgs和Metrics两个对象

        返回值:
        无

        异常描述:
        如果写入文件失败，会抛出RuntimeError异常
        """
        # 从result_pair中提取strategy和metric对象
        strategy, metric = result_pair

        # 提取属性字典
        strategy_dict = asdict(strategy)
        # 过滤掉不需要的属性
        filtered_strategy = {
            k: v 
            for k, v in strategy_dict.items()
            if k not in {"algo", "model", "gbs", "blocks"}
        }
        # 提取metric对象的属性字典
        metric_dict = asdict(metric)

        # 拼接表头（字段名）和字段值
        header = list(filtered_strategy.keys()) + list(metric_dict.keys())
        # 将strategy的属性值转换为字符串
        strategy_values = []
        for k, v in filtered_strategy.items():
            if k == "num_layers_list":
                strategy_values.append(f"[{str(v)}]")
            else:
                strategy_values.append(str(v))
        # 初始化metric的属性值列表
        metric_values = []
        # 遍历metric的属性字典
        for k, v in metric_dict.items():
            # 如果属性名是"tokens_per_npu_per_sec"，则格式化为保留一位小数的字符串
            if k == "tokens_per_npu_per_sec":
                metric_values.append(f"{v:.1f}")
            # 如果属性值是浮点数，则格式化为保留三位小数的字符串
            elif isinstance(v, float):
                metric_values.append(f"{v / 1000:.3f}")
            # 如果属性值是列表或元组，则将每个元素格式化为保留三位小数的字符串，并用逗号连接
            elif isinstance(v, (list, tuple)):
                metric_values.append('[' + ','.join([f"{x / 1000:.3f}" for x in v]) + ']')
        # 将strategy和metric的属性值拼接成一行数据
        line = ','.join(strategy_values + metric_values) + '\n'

        # 定义文件路径
        file_path = f"{self.user_args.log_path}/results.csv"
        try:
            # 检查文件是否存在
            if not os.path.isfile(file_path):
                # 若文件不存在，先写入表头
                with open(file_path, 'w', encoding='utf-8', newline='') as file:
                    file.write(','.join(header) + '\n')
                    file.write(line)
            else:
                # 追加写入数据行
                with open(file_path, 'a', encoding='utf-8', newline='') as file:
                    file.write(line)
        except Exception as e:
            # 如果写入文件失败，抛出异常
            raise RuntimeError(f"写入 results.csv 失败") from e


    def _write_to_sh(self, result_pair, rank, script):
        """
        将tinker并行策略嵌入用户预训练脚本，若没有，则仅生成一个
        :param args: 用户参数
        :param config: tinker搜出的配置
        :param config_rank: 配置排序（按时间）
        :param pretrain_script: 用户的预训练脚本
        :return:
        """
        # 筛出并行策略参数
        strategy, metric = result_pair
        # 格式化输出文件名
        info_text = f'time{metric.time_cost / 1000:.3f}_mem{metric.mem_cost:.2f}'
        split_params = strategy.num_layers_list.replace(',', '_')
        trainsh_path = (
            f"{self.user_args.config_save_path}/{self.user_args.model_name}-{self.user_args.model_size}-rank{rank}"
            f"_seq{self.user_args.seq_length}_tp{strategy.tp}_pp{strategy.pp}_sp{strategy.sp}"
            f"_distopt{strategy.dist_opt}_mbs{strategy.mbs}_gbs{strategy.gbs}_L{split_params}_rc{strategy.recompute}"
            f"_{info_text}.sh")

        script_content = self._write_strategy_to_file(script, strategy)

        # 写文件
        write_lines(script_content, trainsh_path)


    def _print_table(
            self, 
            table_widths: Dict[str, int], 
            result_pairs: List[Tuple[ResultArgs, Metrics]], 
            title: str, 
            save: bool
        ):
        """
        打印日志表格
        :param table_widths: 字典，键为表头名称，值为对应列的宽度
        :param result_pairs: 结果对列表，每个元素为一个元组，包含ResultArgs和Metrics对象
        :param title: 表格标题
        """
        # 打印日志表格
        logger.info('=' * 40 + title + '=' * 40)
        # 获取表头
        headers = list(table_widths.keys())
        # 根据每列宽度生成格式化字符串
        formatter = '|'.join([f'{{:<{width}}}' for width in table_widths.values()])
        # 打印分隔线（根据每列宽度生成 '-' 字符串）
        sep_line = '·'.join(['—' * width for width in table_widths.values()])
        # 首行分割线，特殊处理
        logger.info('·' + sep_line + '·')
        # 打印表头
        logger.info('|' + formatter.format(*headers) + '|')
        logger.info('·' + sep_line + '·')
        # 遍历结果对列表
        for config_rank, result_pair in enumerate(result_pairs):
            # 获取结果字典
            value_dict = self._get_result_dict(config_rank, result_pair)
            # 根据表头获取每行的值
            row_values = [str(value_dict[col]) for col in headers]
            # 打印每行数据
            logger.info('|' + formatter.format(*row_values) + '|')
            if save and self.user_args.detail:
                logger.info('·' + sep_line + '·')
                # TODO:中间结果呈现
                search_args = SearchArgs(**{f.name: getattr(result_pair[0], f.name) for f in fields(SearchArgs)})
                task_param = TaskParam(search_args, result_pair[0].blocks)
                num_layer_list = list(map(int, result_pair[0].num_layers_list.split(',')))
                intervals = convert_to_pp_stage_block_idx(num_layer_list, sum(num_layer_list) + 3)
                self.cost_model.calculate_cost(task_param, intervals, True)
            logger.info('·' + sep_line + '·')
