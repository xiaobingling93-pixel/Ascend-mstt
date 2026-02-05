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

import argparse
import os
from argparse import Namespace
from datetime import datetime, timezone, timedelta
from typing import Union

from tinker.utils.logger import logger
from tinker.utils.utils import project_root


def print_args(args):
    """Print arguments."""
    logger.info('------------------------ arguments ------------------------')
    str_list = []
    for arg in vars(args):
        dots = '.' * (48 - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        logger.info(arg)
    logger.info('-------------------- end of arguments ---------------------')


def preprocess_args(args: argparse.Namespace):
    args.num_npus = args.num_npus_per_node * args.num_nodes

    # 当前固定值
    args.memory_main_params = 2
    args.memory_optimizer = 4

    formatted_time = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d-%H-%M-%S')
    mission_id = (
        f"{args.model_name}-{args.model_size}-gbs{args.global_batch_size}-{args.memory_limit}-{args.num_nodes}"
        f"nodes{args.num_npus_per_node}npus-{formatted_time}")

    if args.pretrain_script_path_search is not None:
        args.pretrain_script_path = args.pretrain_script_path_search

    if args.mode != 'simulate':
        # 结果就落盘在 output_dir
        args.log_path = os.path.join(args.output_dir, mission_id)
        args.config_save_path = os.path.join(args.log_path, 'configs')
        args.log_file = os.path.join(args.log_path, f'{mission_id}.log')

        # 使用 exist_ok=True 参数，这样如果目录已经存在，不会抛出 FileExistsError 异常
        os.makedirs(args.config_save_path, exist_ok=True)

    return args