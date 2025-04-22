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

from msprobe.pytorch.config_check.config_checker import ConfigChecker
from msprobe.pytorch.common.log import logger


def pack(config_filepath):
    ConfigChecker(config_filepath)


def compare(bench_zip_path, cmp_zip_path, outpath):
    ConfigChecker.compare(bench_zip_path, cmp_zip_path, outpath)


def _config_checking_parser(parser):
    parser.add_argument('-pack', '--pack', help='Pack a directory into a zip file')
    parser.add_argument('-c', '--compare', nargs=2, help='Compare two zip files')
    parser.add_argument('-o', '--output', help='output path, default is current directory')


def _run_config_checking_command(args):
    if args.pack:
        pack(args.pack)
    elif args.compare:
        output_dirpath = args.output if args.output else "./config_check_result"
        compare(args.compare[0], args.compare[1], output_dirpath)
    else:
        logger.error("The param is not correct, you need to give '-pack' for pack or '-c' for compare.")
        raise Exception("The param is not correct, you need to give '-pack' for pack or '-c' for compare.")
