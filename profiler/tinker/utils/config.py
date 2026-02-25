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
import argparse
from functools import partial
import json

from tinker.utils.utils import extract_arg_value_from_json, check_path_exist, \
    check_path_before_create, check_files_in_dir, check_file_suffix, project_root
from tinker.utils.logger import logger
from tinker.utils.constant import InitialValues

TINKER_DIR = os.path.join(project_root(), 'tinker')
CONFIG_PATH = os.path.join(TINKER_DIR, 'parameter_config.json')
test_free_args = ['prof_tp', 'prof_sp', 'pretrain_script_path_search']
config_json_parameter = {}
mode = "all"


def parse_args() -> argparse.Namespace:
    """接收命令行参数"""
    # 创建ArgumentParser 对象
    description = 'parse args for tinker auto parallel'
    parser = argparse.ArgumentParser(description=description)

    # 接收人工指定的参数
    parser = add_args(parser)

    # 解析命令行参数
    args = parser.parse_args()
    return args


def add_args(parser: argparse.ArgumentParser):
    # 先判断用户是否传入配置文件路径
    initialize_from_args(parser)
    """定义人工指定参数的函数"""
    # 性能测量(感知器)参数
    add_profile_args(parser)

    # 仿真器参数
    add_simulate_args(parser)

    # 策略寻优参数(优化器)参数, 不包含仿真器参数
    add_search_args(parser)

    return parser


def initialize_from_args(parser):
    global config_json_parameter
    global mode
    global CONFIG_PATH

    # 定义位置参数
    parser.add_argument('-m', '--mode', type=str, default='all', choices=['all', 'profile', 'search', 'simulate'],
                        help='tinker mode, default value is all')
    parser.add_argument('-config', '--config_path', type=str, default=CONFIG_PATH,
                        help='path of parameter_config.json')
    parser.add_argument('--is_full_tune', type=bool, default=False,
                        help='is full parameter finetune')
    # 用户可能修改了config.json的路径
    args, unknown_args = parser.parse_known_args()
    config_path = args.config_path
    mode = args.mode
    config_json_parameter = extract_arg_value_from_json(config_path)


def add_profile_args(parser: argparse.ArgumentParser):
    """添加性能测量(感知器)参数"""
    profile_group = parser.add_argument_group(title='profile information')
    get_profile_arg = partial(get_default_arg, "profile")
    profile_group.add_argument('-name', '--model_name', type=str, default=get_profile_arg("model_name"),
                               help='model name')
    profile_group.add_argument('-size', '--model_size', type=str, default=get_profile_arg("model_size"),
                               help='model size')
    profile_group.add_argument('-sh', '--pretrain_script_path', type=str,
                               default=get_profile_arg("pretrain_script_path"), help='pretrain shell script')
    profile_group.add_argument('-v', '--version', type=str, default=get_profile_arg("version"),
                               help='version for modellink')
    profile_group.add_argument('-p', '--save_path', type=str, default=get_profile_arg("save_path"),
                               help='directory to save profied data, default:`./profiled_data`')
    profile_group.add_argument('-tp', '--prof_tp', type=str, default=get_profile_arg("prof_tp"),
                               help='specify the TP-value for profiling, default for all TP')
    profile_group.add_argument('-sp', '--prof_sp', type=str, default=get_profile_arg("prof_sp"),
                               help='specify the SP-value for profiling, default for all SP')
    profile_group.add_argument('--max_mbs', type=int, default=get_profile_arg("max_mbs"),
                               help='specify the max mbs for profiling, default: 65536')
    profile_group.add_argument('-i', '--task_id', type=str, default=get_profile_arg("task_id"),
                               help='specify suffix of profiled data dir')
    profile_group.add_argument('--max_npu', type=int, default=get_profile_arg("max_npu"),
                               help='specify the max npu-nums, default: 8')


def add_search_args(parser: argparse.ArgumentParser):
    """添加策略寻优参数(优化器)参数, 不包含仿真器参数中的重复项"""
    search_group = parser.add_argument_group(title='search group')
    # 根据mode决定取哪里的值
    global mode
    modified_mode = mode
    if mode == 'all' or mode == 'profile':
        modified_mode = 'search'
    search_group.add_argument('-profiled', '--profiled_data_path', type=str,
                              default=get_default_arg(modified_mode, "profiled_data_path"),
                              help='path of profiled data, required')
    search_group.add_argument('-gbs', '--global_batch_size', type=int,
                              default=get_default_arg(modified_mode, "global_batch_size"),
                              help='global batch size, required')
    search_group.add_argument('-nodes', '--num_nodes', type=int, default=get_default_arg(modified_mode, "num_nodes"),
                              help='number of nodes, required')
    search_group.add_argument('-n', '--num_npus_per_node', type=int,
                              default=get_default_arg(modified_mode, "num_npus_per_node"),
                              help='number of npus on single node, required')
    get_search_arg = partial(get_default_arg, "search")
    search_group.add_argument('-cpus', '--cpus', type=int, default=get_search_arg("cpus"),
                              help='number of cpu, search process will be faster if larger')
    search_group.add_argument('-mem', '--memory_limit', type=int, default=get_search_arg("memory_limit"),
                              help='memory limit')
    search_group.add_argument('-output', '--output_dir', type=str, default=get_search_arg("output_dir"),
                              help='path to save results for optimizer-search, log file etc.')
    search_group.add_argument('-shs', '--pretrain_script_path_search', type=str,
                              default=get_search_arg("pretrain_script_path_search"),
                              help='path to pretrain shell script need to be optimized (defaults to profile phase\'s)')


def add_simulate_args(parser: argparse.ArgumentParser):
    """添加仿真器参数"""
    simulate_group = parser.add_argument_group(title='simulate group')
    get_simulate_arg = partial(get_default_arg, "simulate")
    simulate_group.add_argument('--simu_tp', type=int, default=get_simulate_arg("simu_tp"),
                                help='tensor parallel')
    simulate_group.add_argument('--simu_pp', type=int, default=get_simulate_arg("simu_pp"),
                                help='pipeline parallel')
    simulate_group.add_argument('--simu_ep', type=int, default=get_simulate_arg("simu_ep"),
                                help='expert parallel')
    simulate_group.add_argument('--simu_sp', type=int, default=get_simulate_arg("simu_sp"),
                                help='sequence parallel')
    simulate_group.add_argument('--dist_opt', type=int, default=get_simulate_arg("dist_opt"),
                                help='mode of dist_opt', choices=[0, 1])
    simulate_group.add_argument('-mbs', '--micro_batch_size', type=int, default=get_simulate_arg("micro_batch_size"),
                                help='micro batch size')
    simulate_group.add_argument('--num_layer_list', type=str, default=get_simulate_arg("num_layer_list"),
                                help='a list of number of layers, seperated by comma; e.g., 4,4,4,4, required')
    simulate_group.add_argument('--recompute', type=int, default=get_simulate_arg("recompute"),
                                help='enable full recompute', choices=[0, 1])
    simulate_group.add_argument('-d', '--detail', action='store_true',
                                help='show detailed memory construct')


def get_default_arg(mode_local: str, arg: str):
    default_parameter = InitialValues()
    # 用户赋值则使用用户内容，否则使用默认值
    if mode_local in config_json_parameter:
        if arg in config_json_parameter[mode_local]:
            return config_json_parameter[mode_local][arg]
    return getattr(getattr(default_parameter, mode_local), arg)


def process_path(args):
    if args.mode == 'search':
        args.profiled_data_path = args.profiled_data_path.replace("\\", "/")
        if '/' not in args.profiled_data_path:
            # 文件夹路径入参不含'/'路径分隔符，则认为该文件夹在profiled_data中
            project_dir = project_root()
            args.profiled_data_path = os.path.join(project_dir, 'profiled_data', args.profiled_data_path)


def check_layers(args):
    if args.mode == 'simulate':
        data_path = args.profiled_data_path
        for filename in os.listdir(data_path):
            # 检查文件名称是否是以“model_info”开头的json文件
            if filename.startswith("model_info") and filename.endswith("json"):
                # 构建完整的文件路径
                file_path = os.path.join(data_path, filename)

                # 读取文件内容
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 提取num_layers的值
                    num_layers = data.get("num_layers")
                    parts = args.num_layer_list.split(',')
                    # 把每个部分转成整数
                    int_list = [int(parts) for part in parts]
                    if num_layers != sum(int_list):
                        raise ValueError("sum of num_layer_list should be equal to num_layers")


def check_args(args: argparse.Namespace) -> argparse.Namespace:
    """参数校验"""

    # 检验参数列表各项是否为None
    def check_args_none(args: argparse.Namespace):
        if args.mode == 'profile':
            return
        # all模式下，自动去./profiled_data路径下寻找数据
        if args.profiled_data_path is None and args.mode != 'all':
            raise ValueError("Missing required argument. Please provide --profiled_data_path when running script")
        if args.global_batch_size is None:
            raise ValueError("Missing required argument. Please provide --global_batch_size when running script")

    # 检验路径参数是否有效
    def check_path_valid(mode_local: str):
        """检查路径参数是否有效"""
        if args.mode == 'profile':
            check_path_exist(args.pretrain_script_path)
            check_file_suffix(args.pretrain_script_path, 'sh')
            check_path_before_create(args.save_path)
        elif args.mode == 'simulate':
            check_path_exist(args.profiled_data_path)
            check_files_in_dir(args.profiled_data_path)
        elif args.mode == 'search':
            check_path_exist(args.profiled_data_path)
            check_files_in_dir(args.profiled_data_path)
            check_path_before_create(args.output_dir)
        else:
            check_path_exist(args.pretrain_script_path)
            check_file_suffix(args.pretrain_script_path, 'sh')
            check_path_before_create(args.save_path)
            check_path_before_create(args.output_dir)

    def check_post_train(args: argparse.Namespace):
        if args.is_full_tune:
            return
        if "tune" in args.pretrain_script_path and "full" in args.pretrain_script_path and args.version >= "2.0.0":
            args.is_full_tune = True

    check_args_none(args)
    process_path(args)
    check_layers(args)
    check_path_valid(args.mode)
    check_post_train(args)

    return args
