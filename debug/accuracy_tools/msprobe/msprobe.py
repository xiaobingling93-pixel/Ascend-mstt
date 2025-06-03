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
import importlib.util

from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.core.compare.utils import _compare_parser
from msprobe.core.compare.compare_cli import compare_cli
from msprobe.core.compare.merge_result.merge_result_cli import _merge_result_parser, merge_result_cli
from msprobe.core.config_check.config_check_cli import _config_checking_parser, \
            _run_config_checking_command


def is_module_available(module_name):
    spec = importlib.util.find_spec(module_name)
    return spec is not None


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="msprobe(mindstudio probe), [Powered by MindStudio].\n"
                    "Providing one-site accuracy difference debugging toolkit for training on Ascend Devices.\n"
                    f"For any issue, refer README.md first",
    )

    parser.set_defaults(print_help=parser.print_help)
    parser.add_argument('-f', '--framework', required=True, choices=[Const.PT_FRAMEWORK, Const.MS_FRAMEWORK],
                        help='Deep learning framework.')
    subparsers = parser.add_subparsers()
    subparsers.add_parser('parse')
    compare_cmd_parser = subparsers.add_parser('compare')
    run_ut_cmd_parser = subparsers.add_parser('run_ut')
    multi_run_ut_cmd_parser = subparsers.add_parser('multi_run_ut')
    api_precision_compare_cmd_parser = subparsers.add_parser('api_precision_compare')
    run_overflow_check_cmd_parser = subparsers.add_parser('run_overflow_check')
    code_mapping_cmd_parser = subparsers.add_parser('code_mapping')
    graph_service_cmd_parser = subparsers.add_parser('graph')
    op_generate_cmd_parser = subparsers.add_parser('op_generate')
    merge_result_parser = subparsers.add_parser('merge_result')
    config_checking_parser = subparsers.add_parser('config_check')
    nan_analyze_parser = subparsers.add_parser('nan_analyze')
    _config_checking_parser(config_checking_parser)
    _compare_parser(compare_cmd_parser)
    _merge_result_parser(merge_result_parser)

    is_torch_available = is_module_available("torch")

    if len(sys.argv) < 4:
        parser.print_help()
        sys.exit(0)
    framework_args = parser.parse_args(sys.argv[1:3])
    if framework_args.framework == Const.PT_FRAMEWORK:
        from msprobe.pytorch.api_accuracy_checker.run_ut.run_ut import _run_ut_parser, run_ut_command
        from msprobe.pytorch.parse_tool.cli import parse as cli_parse
        from msprobe.pytorch.api_accuracy_checker.run_ut.multi_run_ut import prepare_config, run_parallel_ut
        from msprobe.pytorch.api_accuracy_checker.compare.api_precision_compare import _api_precision_compare_parser, \
            _api_precision_compare_command
        from msprobe.pytorch.api_accuracy_checker.run_ut.run_overflow_check import _run_overflow_check_parser, \
            _run_overflow_check_command
        from msprobe.visualization.graph_service import _pt_graph_service_parser, _pt_graph_service_command
        from msprobe.pytorch.api_accuracy_checker.generate_op_script.op_generator import _op_generator_parser, \
            _run_operator_generate_commond
        from msprobe.nan_analyze.analyzer import _nan_analyze_parser, _run_nan_analyze

        _run_ut_parser(run_ut_cmd_parser)
        _run_ut_parser(multi_run_ut_cmd_parser)
        multi_run_ut_cmd_parser.add_argument('-n', '--num_splits', type=int, choices=range(1, 65), default=8,
                                             help='Number of splits for parallel processing. Range: 1-64')
        _api_precision_compare_parser(api_precision_compare_cmd_parser)
        _run_overflow_check_parser(run_overflow_check_cmd_parser)
        _pt_graph_service_parser(graph_service_cmd_parser)
        _op_generator_parser(op_generate_cmd_parser)
        _nan_analyze_parser(nan_analyze_parser)
    elif framework_args.framework == Const.MS_FRAMEWORK:
        from msprobe.mindspore.api_accuracy_checker.cmd_parser import add_api_accuracy_checker_argument
        from msprobe.visualization.graph_service import _ms_graph_service_parser, _ms_graph_service_command
        add_api_accuracy_checker_argument(run_ut_cmd_parser)
        from msprobe.mindspore.api_accuracy_checker.cmd_parser import multi_add_api_accuracy_checker_argument
        multi_add_api_accuracy_checker_argument(multi_run_ut_cmd_parser)
        from msprobe.mindspore.code_mapping.cmd_parser import add_ir_parser_arguments
        add_ir_parser_arguments(code_mapping_cmd_parser)

        _ms_graph_service_parser(graph_service_cmd_parser)

        from msprobe.mindspore.api_accuracy_checker.generate_op_script.op_generator import _op_generator_parser, \
            _run_operator_generate_commond
        _op_generator_parser(op_generate_cmd_parser)

    args = parser.parse_args(sys.argv[1:])
    if sys.argv[2] == Const.PT_FRAMEWORK:
        if not is_torch_available:
            logger.error("PyTorch does not exist, please install PyTorch library")
            raise Exception("PyTorch does not exist, please install PyTorch library")
        if sys.argv[3] == "run_ut":
            run_ut_command(args)
        elif sys.argv[3] == "parse":
            cli_parse()
        elif sys.argv[3] == "multi_run_ut":
            config = prepare_config(args)
            run_parallel_ut(config)
        elif sys.argv[3] == "api_precision_compare":
            _api_precision_compare_command(args)
        elif sys.argv[3] == "run_overflow_check":
            _run_overflow_check_command(args)
        elif sys.argv[3] == "graph":
            _pt_graph_service_command(args)
        elif sys.argv[3] == 'op_generate':
            _run_operator_generate_commond(args)
        elif sys.argv[3] == "compare":
            if args.cell_mapping is not None or args.api_mapping is not None:
                logger.error("Argument -cm or -am is not supported in PyTorch framework")
                raise Exception("Argument -cm or -am is not supported in PyTorch framework")
            compare_cli(args)
        elif sys.argv[3] == "merge_result":
            merge_result_cli(args)
        elif sys.argv[3] == "config_check":
            _run_config_checking_command(args)
        elif sys.argv[3] == "nan_analyze":
            _run_nan_analyze(args)
    else:
        if not is_module_available(Const.MS_FRAMEWORK):
            logger.error("MindSpore does not exist, please install MindSpore library")
            raise Exception("MindSpore does not exist, please install MindSpore library")
        if sys.argv[3] == "compare":
            compare_cli(args)
        elif sys.argv[3] == "merge_result":
            merge_result_cli(args)
        elif sys.argv[3] == "run_ut":
            from msprobe.mindspore.api_accuracy_checker.main import api_checker_main
            api_checker_main(args)
        elif sys.argv[3] == "multi_run_ut":
            from msprobe.mindspore.api_accuracy_checker.main import mul_api_checker_main
            mul_api_checker_main(args)
        elif sys.argv[3] == "graph":
            _ms_graph_service_command(args)
        elif sys.argv[3] == 'op_generate':
            _run_operator_generate_commond(args)
        elif sys.argv[3] == "code_mapping":
            from msprobe.mindspore.code_mapping.main import code_mapping_main
            code_mapping_main(args)
        elif sys.argv[3] == "config_check":
            _run_config_checking_command(args)


if __name__ == "__main__":
    main()
