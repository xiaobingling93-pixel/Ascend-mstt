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
from msprobe.core.compare.merge_result.merge_result import merge_result


def parse_list(value):
    """Ensure the input is parsed as a list."""
    if isinstance(value, str):
        return [value]  # 单个字符串转为列表
    elif isinstance(value, list):
        return value  # 已经是列表则返回原值
    raise argparse.ArgumentTypeError("Input must be a list or a single string.")


def _merge_result_parser(parser):
    parser.add_argument("-i", "--input_dir", dest="input_dir", type=str,
                        help="<Required> The compare result path, a dir.", required=True)
    parser.add_argument("-o", "--output_dir", dest="output_dir", type=str,
                        help="<Required> The result merge output path, a dir.", required=True)
    parser.add_argument("-config", "--config-path", dest="config_path", type=str,
                        help="<Required> Yaml path containing distribute APIs and compare indexes for merging data "
                             "from compare results.",
                        required=True)


def merge_result_cli(args):
    merge_result(args.input_dir, args.output_dir, args.config_path)
