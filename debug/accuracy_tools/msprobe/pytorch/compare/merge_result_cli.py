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
from msprobe.core.common.file_utils import FileChecker, create_directory
from msprobe.core.common.const import FileCheckConst
from msprobe.pytorch.compare.merge_result import merge_result


def parse_list(value):
    """Ensure the input is parsed as a list."""
    if isinstance(value, str):
        return [value]  # 单个字符串转为列表
    elif isinstance(value, list):
        return value  # 已经是列表则返回原值
    raise argparse.ArgumentTypeError("Input must be a list or a single string.")


def _merge_result_parser(parser):
    parser.add_argument("-i", "--input_dir", dest="input_dir", type=str,
                        help="<Required> The compare input path, a dir.", required=True)
    parser.add_argument("-o", "--output_dir", dest="output_dir", type=str,
                        help="<Required> The compare task result out path.", required=True)
    parser.add_argument("-api", "--api-yaml-path", dest="api_yaml_path", type=str,
                        help="<Required> Yaml path containing distribute APIs for merging data from compare results.",
                        required=True)
    parser.add_argument("-index", '--compare-index-list', dest="compare_index_list",
                        choices=["Max diff", "Min diff", "Mean diff", "L2norm diff",
                                 "Consine", "MaxAbsError", "MaxRelativeErr",
                                 "One Thousandth Err Ratio", "Five Thousandth Err Ratio"],
                        nargs="*",
                        type=parse_list,  # 确保解析为列表
                        default=[],  # 支持零个或多个值, 默认值为空列表
                        help="<optional> Compare indexes to merge, default is []."
                        )


def merge_result_cli(args):
    input_dir_checker = FileChecker(args.input_dir, FileCheckConst.DIR, FileCheckConst.READ_ABLE)     # TODO 入参需要确认
    input_dir = input_dir_checker.common_check()
    create_directory(args.output_dir)

    merge_result(input_dir, args.output_dir, args.api_yaml_path, args.compare_index_list)
