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

import os

from msprobe.core.common.file_utils import check_file_or_directory_path, create_directory, FileChecker


def add_ir_parser_arguments(parser):
    parser.add_argument('--ir', type=str, required=True, help="Path to the graph file")
    parser.add_argument('--dump_data', type=str, required=True, default=None, help="Path to data dir")
    parser.add_argument('--output', type=str, required=False, default="./", help="Path to output dir")


def check_args(args):
    args.ir = os.path.abspath(args.ir)

    check_file_or_directory_path(args.ir)

    args.dump_data = os.path.abspath(args.dump_data)
    if os.path.isdir(args.dump_data):
        check_file_or_directory_path(args.dump_data, isdir=True)
    else:
        check_file_or_directory_path(args.dump_data, isdir=False)

    args.output = os.path.abspath(args.output)
    create_directory(args.output)

