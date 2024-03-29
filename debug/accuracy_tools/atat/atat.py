# Copyright (c) 2024, Huawei Technologies Co., Ltd.
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
import sys
from api_accuracy_checker.run_ut.run_ut import _run_ut_parser, run_ut_command
from ptdbg_ascend.src.python.ptdbg_ascend.parse_tool.cli import parse as cli_parse
from api_accuracy_checker.run_ut.multi_run_ut import prepare_config, run_parallel_ut
from api_accuracy_checker.compare.api_precision_compare import _api_precision_compare_parser, _api_precision_compare_command
from api_accuracy_checker.run_ut.run_overflow_check import _run_overflow_check_parser, _run_overflow_check_command


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="atat(ascend training accuracy tools), [Powered by MindStudio].\n"
        "Providing one-site accuracy difference debugging toolkit for training on Ascend Devices.\n"
        f"For any issue, refer README.md first",
    )
    parser.set_defaults(print_help=parser.print_help)
    subparsers = parser.add_subparsers()
    subparsers.add_parser('parse')
    run_ut_cmd_parser = subparsers.add_parser('run_ut')
    multi_run_ut_cmd_parser = subparsers.add_parser('multi_run_ut')
    api_precision_compare_cmd_parser = subparsers.add_parser('api_precision_compare')
    run_overflow_check_cmd_parser = subparsers.add_parser('run_overflow_check')
    _run_ut_parser(run_ut_cmd_parser)
    _run_ut_parser(multi_run_ut_cmd_parser)
    multi_run_ut_cmd_parser.add_argument('-n', '--num_splits', type=int, choices=range(1, 65), default=8,
                                         help='Number of splits for parallel processing. Range: 1-64')
    _api_precision_compare_parser(api_precision_compare_cmd_parser)
    _run_overflow_check_parser(run_overflow_check_cmd_parser)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args(sys.argv[1:])
    if sys.argv[1] == "run_ut":
        run_ut_command(args)
    elif sys.argv[1] == "parse":
        cli_parse()
    elif sys.argv[1] == "multi_run_ut":
        config = prepare_config(args)
        run_parallel_ut(config)
    elif sys.argv[1] == "api_precision_compare":
        _api_precision_compare_command(args)
    elif sys.argv[1] == "run_overflow_check":
        _run_overflow_check_command(args)


if __name__ == "__main__":
    main()