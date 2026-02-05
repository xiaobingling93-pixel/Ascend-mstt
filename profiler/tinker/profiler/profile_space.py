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

# 用于动态修剪profile范围，避免在知道一个配置会爆内存之后继续搜肯定会继续爆内存的配置
import argparse
import itertools
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
import datetime
from typing import List, Optional, Tuple

from tinker.profiler import gen_model_structure
from tinker.profiler.profile_classes import ScriptArgs
from tinker.model.adapter_utils import gen_block_adapter
from tinker.utils.config import TINKER_DIR
from tinker.utils.logger import init_log, logger
from tinker.utils.utils import extract_and_format_model_size
from tinker.version import dump_task_info


def ensure_dir_exists(file_path, pre_logging_text):
    """
    确保文件路径存在，如果不存在则创建所需的目录
    """
    # 获取文件目录
    dir_name = os.path.dirname(file_path)

    # 检查目录是否存在
    if not os.path.exists(dir_name):
        # 创建目录
        os.makedirs(dir_name)
        pre_logging_text.append(f"Created directory: {dir_name}")


class MemoryImpact:
    """这里的影响是对是对内存瓶颈的影响，不是内存绝对值"""
    REDUCE = 1  # 表示参数增大可以减少内存开销
    NEUTRAL = 0
    INCREASE = -1

    @staticmethod
    def _opposite(impact):
        return -impact

    @classmethod
    def judge(cls, v: int, v_base: int, impact: int) -> int:
        if impact == cls.NEUTRAL or v_base == v:
            return cls.NEUTRAL
        if v > v_base:
            return impact
        return cls._opposite(impact)


@dataclass
class Feature:
    name: str
    range: List[int]
    memory_impact: int


class ArgSpace:
    PENDING = -1
    FAILED = 1
    USELESS = 0

    def __init__(self, features: List[Feature], num_npu: int, max_mbs, model_args) -> None:
        self._finished = False
        self.features = {param.name: param for param in features}
        self._max_npu = num_npu
        self._configs = None  # type: Optional[List[ScriptArgs, List]]
        self._max_mbs = max_mbs
        self._max_mbs_list = None
        self._model_args = model_args
        self._generate_configs()

    def fresh_config(self, config_index, max_mbs: int):
        """config是可以运行的配置，所以拿掉这个config的阻塞"""
        self._max_mbs_list[config_index] = max_mbs

    def get_config(self) -> Tuple[int, Optional[ScriptArgs], int]:
        """返回一个没有前序阻塞是待运行或者失败的配置"""
        if self._finished:
            return -1, None, 0
        for i, status in enumerate(self._max_mbs_list):
            if status != self.PENDING:
                continue
            config, block_list = self._configs[i]
            max_mbs = self._max_mbs
            for index in block_list:
                if self._max_mbs_list[index] == self.FAILED or self._max_mbs_list[index] == self.USELESS:
                    self._max_mbs_list[i] = self.USELESS
                    break
                if self._max_mbs_list[index] == self.PENDING:
                    break
                if self._max_mbs_list[index] < max_mbs:
                    max_mbs = self._max_mbs_list[index]
            else:
                return i, config, max_mbs
        else:
            self._finished = True
            return -1, None, 0

    def _generate_configs(self):
        # 生成参数组合的笛卡尔积
        param_names = [param.name for param in self.features.values()]
        param_ranges = [param.range for param in self.features.values()]
        self._configs = []
        for combination in itertools.product(*param_ranges):
            script_config = ScriptArgs(*combination)  # 这样要手动对齐，是不是还是用**dict
            if script_config.is_legal(self._max_npu, self._model_args):
                self._configs.append([script_config, []])  # 配置 阻塞列表

        self._max_mbs_list = [self.PENDING] * len(self._configs)
        # 生成阻塞
        for index1, (config1, blocks1) in enumerate(self._configs[:-1]):
            for index2, (config2, blocks2) in enumerate(self._configs[index1 + 1:]):
                impact = self._mem_impact(config1, config2)
                if impact == MemoryImpact.REDUCE:
                    blocks2.append(index1)
                elif impact == MemoryImpact.INCREASE:
                    blocks1.append(index2 + index1 + 1)

    def _mem_impact(self, config: ScriptArgs, config_base: ScriptArgs) -> int:
        impact_set = set()
        for k, v in config.items():
            v_base = getattr(config_base, k)
            impact = MemoryImpact.judge(v, v_base, self.features[k].memory_impact)
            if impact != MemoryImpact.NEUTRAL:
                impact_set.add(impact)
                if len(impact_set) > 1:
                    return MemoryImpact.NEUTRAL
        if MemoryImpact.REDUCE in impact_set:
            # 内存影响正向，阻塞
            return MemoryImpact.REDUCE
        # 反之，被阻塞
        return MemoryImpact.INCREASE


def pre_log(pre_logging_text):
    # 打印生成logger阶段待输出的内容
    for text in pre_logging_text:
        logger.info(text)


class TinkerScripter:
    MAX_MBS = 2 ** 30
    MAX_NPU = 8
    TP = None
    SP = None
    EP = None

    def __init__(self, model, model_size, suffix, save_path, version, model_args, is_full_tune):
        self.model = model
        self.model_size = model_size
        self.profiler_script = f"{TINKER_DIR}/profiler/profile.sh"
        self.suffix = suffix
        self.save_path = save_path
        self.version = version
        self.model_args = model_args
        self.is_full_tune = '1' if is_full_tune else '0'

    @property
    def can_ep(self):
        return hasattr(self.model_args, 'num_experts')

    @staticmethod
    def post_process(mbs_limit, oom, process, torchrun_failed):
        try:
            stderr = process.communicate(timeout=100)[1]
        except subprocess.TimeoutExpired:
            process.kill()
            stderr = None
        if stderr:
            logger.info(f"stderr: {stderr}")
        if process.returncode:
            if oom:
                logger.info(f"profile内存溢出于{mbs_limit}，将裁剪剩余并行策略探索空间")
            elif torchrun_failed:
                logger.warning(f"torchrun执行错误")
            else:
                logger.warning(f"脚本执行错误")

    @staticmethod
    def is_valid_value(value, space):
        return space is None or value in space

    def get_arg_space(self):
        tp_space = [i for i in range(1, self.MAX_NPU + 1) if self.is_valid_value(i, self.TP)]
        sp_space = [i for i in range(2) if self.is_valid_value(i, self.SP)]
        ep_space = [i for i in range(1, self.MAX_NPU + 1)] if self.can_ep else [1]
        return ArgSpace([
            Feature(name="tp", range=tp_space, memory_impact=MemoryImpact.REDUCE),
            Feature(name="sp", range=sp_space, memory_impact=MemoryImpact.REDUCE),
            Feature(name="ep", range=ep_space, memory_impact=MemoryImpact.REDUCE),
        ], self.MAX_NPU, self.MAX_MBS, self.model_args)

    def run_config(self, config: ScriptArgs, max_mbs: int) -> int:
        # 格式化model_size
        model_size = extract_and_format_model_size(self.model_size)
        # 测量block内并行策略性能 tp sp ep
        command = ['bash', self.profiler_script, self.model, model_size, *config.cmd_text_list, str(max_mbs),
                   self.save_path, self.suffix, self.version, self.is_full_tune]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   preexec_fn=os.setsid)
        mbs_limit = self.MAX_MBS
        mbs_now = 1
        torchrun_failed = False
        oom = False
        already_killed = False
        output_stack = ''
        # 实时读取输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output = output.strip('\n')
                if len(output) == 1:
                    output_stack += output
                    continue
                if output_stack:
                    logger.info(output_stack)
                    output_stack = ''
                logger.info(output)

                # 随地记录mbs
                if "mbs = " in output:
                    mbs_now = int(re.search(r'mbs = (\d+)', output)[1])

                # profile下爆显存，直接终止
                if "NPU out of memory" in output and not already_killed:
                    mbs_limit = mbs_now
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    already_killed = True
                    logger.info(f'OOM when mbs is {mbs_now}')
                    oom = True

                if "[Tinker-Profiler] OOM when mbs" in output:
                    mbs_limit = int(re.search(r'\[Tinker-Profiler] OOM when mbs=(\d+)', output)[1])

            if '.py FAILED' in output and not already_killed:
                logger.info(f'COMMAND {command} FAILED')
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                torchrun_failed = True

        # 获取剩余的标准错误输出
        self.post_process(mbs_limit, oom, process, torchrun_failed)

        return mbs_limit


def get_model_structure(args):
    # 转换训练脚本
    sys.argv[1:] = ['-p', args.pretrain_script_path, '-m', args.model_name, '-s', args.model_size, '-o']
    model_cmd = gen_model_structure.main(from_sh=False)
    model_args = argparse.Namespace()
    for key, value in model_cmd:
        if isinstance(value, int) or value.isdigit():
            setattr(model_args, key, int(value))
        elif not value:
            setattr(model_args, key, True)
    model_args.group_query_attention = getattr(model_args, 'group_query_attention', False)
    model_args.num_query_groups = getattr(model_args, 'num_query_groups', 1)
    if args.max_npu is None:
        TinkerScripter.MAX_NPU = model_args.nproc_per_node
    # 删除用于传递单节点卡数的属性，该属性并不是模型参数
    del model_args.nproc_per_node

    return model_args


def profile_inter_block(save_path):
    """若父文件夹无`p2p_intra_node.csv`，则测量后同时置于父文件夹和`save_path`，否则直接将已存在的文件放在`save_path`中"""
    parent_dir = os.path.dirname(save_path)
    file_name = 'p2p_intra_node.csv'
    # 检查父目录中是否存在p2p_intra_node.csv
    csv_path = os.path.join(parent_dir, file_name)
    if os.path.isfile(csv_path):
        shutil.copy(csv_path, save_path)
    else:
        profile_cmd = ['bash', f"{TINKER_DIR}/profiler/localhost_profile_intra_node_p2p.sh", save_path]
        subprocess.run(profile_cmd)
        csv_path = os.path.join(save_path, file_name)
        shutil.copy(csv_path, parent_dir)


def run(args: argparse.Namespace):
    if args.mode != 'all' and args.mode != 'profile':
        return None
    # 初始化
    start_time = time.time()
    model_name = args.model_name
    model_size = args.model_size
    TinkerScripter.MAX_NPU = args.max_npu
    # (待改进)将TinkerScripter.MAX_MBS逻辑更换为可取的最大mbs取值（当前取不到）
    TinkerScripter.MAX_MBS = args.max_mbs + 1
    if args.prof_tp is not None:
        TinkerScripter.TP = set(map(int, args.prof_tp.split(',')))
    if args.prof_sp is not None:
        TinkerScripter.SP = set(map(int, args.prof_sp.split(',')))

    pre_logging_text = []
    tz = datetime.timezone(datetime.timedelta(hours=8))
    suffix = datetime.datetime.now(tz=tz).strftime("%y%m%d-%H%M%S")
    if args.task_id:
        suffix += f'-{args.task_id}'
    task_name = f'profiled-data-{model_name}-{model_size}-{suffix}'
    dir_path = os.path.join(args.save_path, task_name)
    args.profiled_data_path = dir_path
    # 初始化 logger
    abs_dir_path = os.path.abspath(dir_path)
    log_file = os.path.join(abs_dir_path, 'script.log')
    ensure_dir_exists(log_file, pre_logging_text)
    init_log(log_file, logging.INFO)
    dump_task_info(dir_path, vars(args))
    # (待改进)规划客户使用时的suffix使用逻辑
    pre_log(pre_logging_text)
    model_args = get_model_structure(args)
    # 自动生成adapter
    gen_block_adapter(hasattr(model_args, 'use_mcore_models') and model_args.use_mcore_models)
    profiler = TinkerScripter(model_name, model_size, suffix, args.save_path, args.version, model_args,
                              args.is_full_tune)
    # 生成参数空间
    arg_space = profiler.get_arg_space()
    # 迭代获取config
    index, config, max_mbs = arg_space.get_config()
    # 刷新configs
    while index >= 0:
        mbs_limit = profiler.run_config(config, max_mbs)
        arg_space.fresh_config(index, mbs_limit)
        mbs_hint = 'DONE' if mbs_limit == profiler.MAX_MBS else f'mbs < {mbs_limit}'
        logger.info(f'{index} {config} -> {mbs_hint}')
        # 再拿一个
        index, config, max_mbs = arg_space.get_config()
    profile_inter_block(dir_path)
    # 输出profile情况
    logger.info(f'Profile Space Total Time: {time.time() - start_time:.2f}')
    logger.info(f'Profile Data Saved at {dir_path}')
    return dir_path
