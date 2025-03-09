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

import os
import time
import json
from msprobe.core.common.file_utils import (check_file_type, create_directory, FileChecker,
                                            check_file_or_directory_path, load_json)
from msprobe.core.common.const import FileCheckConst, Const
from msprobe.core.common.utils import CompareException
from msprobe.core.overflow_check.checker import AnomalyDetector
from msprobe.visualization.compare.graph_comparator import GraphComparator
from msprobe.visualization.utils import GraphConst, check_directory_content
from msprobe.visualization.builder.graph_builder import GraphBuilder, GraphExportConfig
from msprobe.core.common.log import logger
from msprobe.visualization.graph.node_colors import NodeColors
from msprobe.core.compare.layer_mapping import generate_api_mapping_by_layer_mapping
from msprobe.core.compare.utils import check_and_return_dir_contents
from msprobe.visualization.graph.distributed_analyzer import DistributedAnalyzer

current_time = time.strftime("%Y%m%d%H%M%S")


def _compare_graph(input_param, args):
    logger.info('Start building model graphs...')
    # 对两个数据进行构图
    dump_path_n = input_param.get('npu_path')
    dump_path_b = input_param.get('bench_path')
    construct_path_n = FileChecker(os.path.join(dump_path_n, GraphConst.CONSTRUCT_FILE),
                                   FileCheckConst.FILE, FileCheckConst.READ_ABLE).common_check()
    construct_path_b = FileChecker(os.path.join(dump_path_b, GraphConst.CONSTRUCT_FILE),
                                   FileCheckConst.FILE, FileCheckConst.READ_ABLE).common_check()
    data_path_n = FileChecker(os.path.join(dump_path_n, GraphConst.DUMP_FILE), FileCheckConst.FILE,
                              FileCheckConst.READ_ABLE).common_check()
    data_path_b = FileChecker(os.path.join(dump_path_b, GraphConst.DUMP_FILE), FileCheckConst.FILE,
                              FileCheckConst.READ_ABLE).common_check()
    stack_path_n = FileChecker(os.path.join(dump_path_n, GraphConst.STACK_FILE), FileCheckConst.FILE,
                               FileCheckConst.READ_ABLE).common_check()
    stack_path_b = FileChecker(os.path.join(dump_path_b, GraphConst.STACK_FILE), FileCheckConst.FILE,
                               FileCheckConst.READ_ABLE).common_check()
    graph_n = GraphBuilder.build(construct_path_n, data_path_n, stack_path_n, complete_stack=args.complete_stack)
    graph_b = GraphBuilder.build(construct_path_b, data_path_b, stack_path_b, complete_stack=args.complete_stack)
    logger.info('Model graphs built successfully, start Comparing graphs...')
    # 基于graph、stack和data进行比较
    dump_path_param = {
        'npu_json_path': data_path_n,
        'bench_json_path': data_path_b,
        'stack_json_path': stack_path_n,
        'is_print_compare_log': input_param.get("is_print_compare_log", True)
    }
    mapping_dict = None
    if args.layer_mapping:
        yaml_path = FileChecker(args.layer_mapping, FileCheckConst.FILE, FileCheckConst.READ_ABLE).common_check()
        try:
            mapping_dict = generate_api_mapping_by_layer_mapping(data_path_n, data_path_b, yaml_path)
        except Exception:
            logger.warning('The layer mapping file parsing failed, please check file format, mapping is not effective.')
    graph_comparator = GraphComparator([graph_n, graph_b], dump_path_param, args, mapping_dict=mapping_dict)
    graph_comparator.compare()
    micro_steps = graph_n.paging_by_micro_step(graph_b)
    # 开启溢出检测
    if args.overflow_check:
        graph_n.overflow_check()
        graph_b.overflow_check()

    return CompareGraphResult(graph_n, graph_b, graph_comparator, micro_steps)


def _export_compare_graph_result(args, graphs, graph_comparator, micro_steps,
                                 output_file_name=f'compare_{current_time}.vis'):
    create_directory(args.output_path)
    output_path = os.path.join(args.output_path, output_file_name)
    task = GraphConst.GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING.get(graph_comparator.ma.compare_mode)
    export_config = GraphExportConfig(graphs[0], graphs[1], graph_comparator.ma.get_tool_tip(),
                                      NodeColors.get_node_colors(graph_comparator.ma.compare_mode), micro_steps, task,
                                      args.overflow_check)
    GraphBuilder.to_json(output_path, export_config)
    logger.info(f'Model graphs compared successfully, the result file is saved in {output_path}')


def _build_graph(dump_path, args):
    logger.info('Start building model graph...')
    construct_path = FileChecker(os.path.join(dump_path, GraphConst.CONSTRUCT_FILE), FileCheckConst.FILE,
                                 FileCheckConst.READ_ABLE).common_check()
    data_path = FileChecker(os.path.join(dump_path, GraphConst.DUMP_FILE), FileCheckConst.FILE,
                            FileCheckConst.READ_ABLE).common_check()
    stack_path = FileChecker(os.path.join(dump_path, GraphConst.STACK_FILE), FileCheckConst.FILE,
                             FileCheckConst.READ_ABLE).common_check()
    graph = GraphBuilder.build(construct_path, data_path, stack_path, complete_stack=args.complete_stack)
    micro_steps = graph.paging_by_micro_step()
    # 开启溢出检测
    if args.overflow_check:
        graph.overflow_check()
    return BuildGraphResult(graph, micro_steps)


def _export_build_graph_result(out_path, graph, micro_steps, overflow_check,
                               output_file_name=f'build_{current_time}.vis'):
    create_directory(out_path)
    output_path = os.path.join(out_path, output_file_name)
    GraphBuilder.to_json(output_path, GraphExportConfig(graph, micro_steps=micro_steps, overflow_check=overflow_check))
    logger.info(f'Model graph built successfully, the result file is saved in {output_path}')


def _compare_graph_ranks(input_param, args, step=None):
    dump_rank_n = input_param.get('npu_path')
    dump_rank_b = input_param.get('bench_path')
    npu_ranks = sorted(check_and_return_dir_contents(dump_rank_n, Const.RANK))
    bench_ranks = sorted(check_and_return_dir_contents(dump_rank_b, Const.RANK))
    if npu_ranks != bench_ranks:
        logger.error('The number of ranks in the two runs are different. Unable to match the ranks.')
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    compare_graph_results = []
    for nr, br in zip(npu_ranks, bench_ranks):
        logger.info(f'Start processing data for {nr}...')
        input_param['npu_path'] = os.path.join(dump_rank_n, nr)
        input_param['bench_path'] = os.path.join(dump_rank_b, br)
        output_file_name = f'compare_{step}_{nr}_{current_time}.vis' if step else f'compare_{nr}_{current_time}.vis'
        result = _compare_graph(input_param, args)
        result.output_file_name = output_file_name
        if nr != Const.RANK:
            try:
                result.rank = int(nr.replace(Const.RANK, ""))
            except Exception as e:
                logger.error('The folder name format is incorrect, expected rank+number.')
                raise CompareException(CompareException.INVALID_PATH_ERROR) from e
        # 暂存所有rank的graph，用于匹配rank间的分布式节点
        compare_graph_results.append(result)

    # 匹配rank间的分布式节点
    if len(compare_graph_results) > 1:
        DistributedAnalyzer({obj.rank: obj.graph_n for obj in compare_graph_results},
                            args.overflow_check).distributed_match()
        DistributedAnalyzer({obj.rank: obj.graph_b for obj in compare_graph_results},
                            args.overflow_check).distributed_match()

    for result in compare_graph_results:
        _export_compare_graph_result(args, [result.graph_n, result.graph_b], result.graph_comparator,
                                     result.micro_steps, output_file_name=result.output_file_name)


def _compare_graph_steps(input_param, args):
    dump_step_n = input_param.get('npu_path')
    dump_step_b = input_param.get('bench_path')

    npu_steps = sorted(check_and_return_dir_contents(dump_step_n, Const.STEP))
    bench_steps = sorted(check_and_return_dir_contents(dump_step_b, Const.STEP))

    if npu_steps != bench_steps:
        logger.error('The number of steps in the two runs are different. Unable to match the steps.')
        raise CompareException(CompareException.INVALID_PATH_ERROR)

    for folder_step in npu_steps:
        logger.info(f'Start processing data for {folder_step}...')
        input_param['npu_path'] = os.path.join(dump_step_n, folder_step)
        input_param['bench_path'] = os.path.join(dump_step_b, folder_step)

        _compare_graph_ranks(input_param, args, step=folder_step)


def _build_graph_ranks(dump_ranks_path, args, step=None):
    ranks = sorted(check_and_return_dir_contents(dump_ranks_path, Const.RANK))
    build_graph_results = []
    for rank in ranks:
        logger.info(f'Start processing data for {rank}...')
        dump_path = os.path.join(dump_ranks_path, rank)
        output_file_name = f'build_{step}_{rank}_{current_time}.vis' if step else f'build_{rank}_{current_time}.vis'
        result = _build_graph(dump_path, args)
        result.output_file_name = output_file_name
        if rank != Const.RANK:
            try:
                result.rank = int(rank.replace(Const.RANK, ""))
            except Exception as e:
                logger.error('The folder name format is incorrect, expected rank+number.')
                raise CompareException(CompareException.INVALID_PATH_ERROR) from e
        build_graph_results.append(result)

    if len(build_graph_results) > 1:
        DistributedAnalyzer({obj.rank: obj.graph for obj in build_graph_results},
                            args.overflow_check).distributed_match()

    for result in build_graph_results:
        _export_build_graph_result(args.output_path, result.graph, result.micro_steps, args.overflow_check,
                                   result.output_file_name)


def _build_graph_steps(dump_steps_path, args):
    steps = sorted(check_and_return_dir_contents(dump_steps_path, Const.STEP))
    for step in steps:
        logger.info(f'Start processing data for {step}...')
        dump_ranks_path = os.path.join(dump_steps_path, step)
        _build_graph_ranks(dump_ranks_path, args, step)


def _graph_service_parser(parser):
    parser.add_argument("-i", "--input_path", dest="input_path", type=str,
                        help="<Required> The compare input path, a dict json.", required=True)
    parser.add_argument("-o", "--output_path", dest="output_path", type=str,
                        help="<Required> The compare task result out path.", required=True)
    parser.add_argument("-lm", "--layer_mapping", dest="layer_mapping", type=str,
                        help="<Optional> The layer mapping file path.", required=False)
    parser.add_argument("-oc", "--overflow_check", dest="overflow_check", action="store_true",
                        help="<Optional> whether open overflow_check for graph.", required=False)
    parser.add_argument("-f", "--fuzzy_match", dest="fuzzy_match", action="store_true",
                        help="<Optional> Whether to perform a fuzzy match on the api name.", required=False)
    parser.add_argument("-cs", "--complete_stack", dest="complete_stack", action="store_true",
                        help="<Optional> Whether to use complete stack information.", required=False)


def _graph_service_command(args):
    input_param = load_json(args.input_path)
    npu_path = input_param.get("npu_path")
    bench_path = input_param.get("bench_path")
    check_file_or_directory_path(npu_path, isdir=True)
    if bench_path:
        check_file_or_directory_path(bench_path, isdir=True)
    if check_file_type(npu_path) == FileCheckConst.DIR and not bench_path:
        content = check_directory_content(npu_path)
        if content == GraphConst.RANKS:
            _build_graph_ranks(npu_path, args)
        elif content == GraphConst.STEPS:
            _build_graph_steps(npu_path, args)
        else:
            result = _build_graph(npu_path, args)
            _export_build_graph_result(args.output_path, result.graph, result.micro_steps, args.overflow_check)
    elif check_file_type(npu_path) == FileCheckConst.DIR and check_file_type(bench_path) == FileCheckConst.DIR:
        content_n = check_directory_content(npu_path)
        content_b = check_directory_content(bench_path)
        if content_n != content_b:
            raise ValueError('The directory structures of npu_path and bench_path are inconsistent.')
        if content_n == GraphConst.RANKS:
            _compare_graph_ranks(input_param, args)
        elif content_n == GraphConst.STEPS:
            _compare_graph_steps(input_param, args)
        else:
            result = _compare_graph(input_param, args)
            _export_compare_graph_result(args, [result.graph_n, result.graph_b],
                                         result.graph_comparator, result.micro_steps)
    else:
        logger.error("The npu_path or bench_path should be a folder.")
        raise CompareException(CompareException.INVALID_COMPARE_MODE)


def _pt_graph_service_parser(parser):
    _graph_service_parser(parser)


def _pt_graph_service_command(args):
    _graph_service_command(args)


def _ms_graph_service_parser(parser):
    _graph_service_parser(parser)


def _ms_graph_service_command(args):
    _graph_service_command(args)


class CompareGraphResult:
    def __init__(self, graph_n, graph_b, graph_comparator, micro_steps, rank=0, output_file_name=''):
        self.graph_n = graph_n
        self.graph_b = graph_b
        self.graph_comparator = graph_comparator
        self.micro_steps = micro_steps
        self.rank = rank
        self.output_file_name = output_file_name


class BuildGraphResult:
    def __init__(self, graph, micro_steps, rank=0, output_file_name=''):
        self.graph = graph
        self.micro_steps = micro_steps
        self.rank = rank
        self.output_file_name = output_file_name
