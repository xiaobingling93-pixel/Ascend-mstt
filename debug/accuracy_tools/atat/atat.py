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
from api_accuracy_checker.run_ut_run_ut import _run_ut_parser, run_ut_command

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="atat(Ascend Training Accuracy Tools), [Powered by MindStudio].\n"
        "Providing one-site accuracy difference debugging toolkit for training on Ascend Devices.\n"
        f"For any issue, refer README first TODO",
    )
    subparsers = parser.add_subparsers(help='commands')
    #parse_cmd_parser = subparsers.add_parser('parse', help='[TODO]parse command')
    run_ut_cmd_parser = subparsers.add_parser('run_ut', help='[TODO]run_ut command')
    #multi_run_ut_cmd_parser =  subparsers.add_parser('parse', help='[TODO]multi_run_ut command')
    #benchmark_compare_cmd_parser =  subparsers.add_parser('parse', help='[TODO]benchmark_compare command')
    parser.set_defaults(print_help=parser.print_help)
    run_parser = _run_ut_parser(run_ut_cmd_parser)
    args = parser.parse_args(sys.argv[1:])
    if sys.argv[1] == "run_ut":
        run_ut_command(args)


if __name__ == "__main__":
    main()