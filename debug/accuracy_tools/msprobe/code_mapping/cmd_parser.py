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

import argparse
import os
import logging
from pathlib import Path


from msprobe.core.common.file_utils import check_file_or_directory_path, create_directory, FileChecker
from msprobe.core.common.utils import Const, MsprobeBaseException
from msprobe.mindspore.common.log import logger


def add_ir_parser_arguments(parser):
    parser.add_argument('--ir', type=str, required=True, help="Path to the graph file")
    parser.add_argument('--data', type=str, required=False, default=None, help="Path to data dir")
    parser.add_argument('--output', type=str, required=False, default="./", help="Path to output dir")
    #生成mapping用时间戳


def check_args(args):
    """
    对命令行参数进行安全校验。

    Parameters:
        args (argparse.Namespace): 解析后的命令行参数。

    Raises:
        MsprobeBaseException: 当参数校验失败时抛出。
    """

    # 校验 --ir 参数
    ir_checker = FileChecker(
        file_path=args.ir,
        path_type=FileCheckConst.FILE,
        ability=FileCheckConst.READ_ABLE,
    )
    args.ir = ir_checker.common_check()

    # 校验 --data 参数（如果提供）
    if args.data:
        data_checker = FileChecker(
            file_path=args.data,
            path_type=FileCheckConst.DIR,
            ability=FileCheckConst.READ_ABLE
        )
        args.data = data_checker.common_check()

    # 校验 --output 参数
    output_checker = FileChecker(
        file_path=args.output,
        path_type=FileCheckConst.DIR,
        ability=FileCheckConst.WRITE_ABLE
    )
    args.output = output_checker.common_check()
    create_directory(args.output)

