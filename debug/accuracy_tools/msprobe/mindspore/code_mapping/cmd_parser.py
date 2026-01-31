# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


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

