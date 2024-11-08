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
from msprobe.core.common.file_utils import FileOpen, check_file_type, create_directory, FileChecker
from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.utils import CompareException
from msprobe.visualization.compare.graph_comparator import GraphComparator
from msprobe.visualization.utils import GraphConst
from msprobe.visualization.builder.graph_builder import GraphBuilder, GraphExportConfig
from msprobe.core.common.log import logger
from msprobe.visualization.mapping_config import MappingConfig, MappingInfo, DATA_MAPPING, LAYER_MAPPING
from msprobe.visualization.graph.node_colors import NodeColors

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
    graph_n = GraphBuilder.build(construct_path_n, data_path_n)
    graph_b = GraphBuilder.build(construct_path_b, data_path_b)
    logger.info('Model graphs built successfully, start Comparing graphs...')
    # 基于graph、stack和data进行比较
    stack_path = FileChecker(os.path.join(dump_path_n, GraphConst.STACK_FILE), FileCheckConst.FILE,
                             FileCheckConst.READ_ABLE).common_check()
    dump_path_param = {
        'npu_json_path': data_path_n,
        'bench_json_path': data_path_b,
        'stack_json_path': stack_path,
        'is_print_compare_log': input_param.get("is_print_compare_log", True)
    }
    mapping_config = None
    if args.data_mapping:
        mapping_config = MappingConfig(args.data_mapping, MappingInfo(DATA_MAPPING))
    elif args.layer_mapping:
        mapping_config = MappingConfig(args.layer_mapping, MappingInfo(LAYER_MAPPING, data_path_n, data_path_b))
    graph_comparator = GraphComparator([graph_n, graph_b], dump_path_param, args.output_path, args.framework,
                                       mapping_config=mapping_config)
    graph_comparator.compare()
    micro_steps = graph_n.paging_by_micro_step(graph_b)
    create_directory(args.output_path)
    output_path = os.path.join(args.output_path, f'compare_{current_time}.vis')
    export_config = GraphExportConfig(graph_n, graph_b, graph_comparator.ma.get_tool_tip(),
                                      NodeColors.get_node_colors(graph_comparator.ma.compare_mode), micro_steps)
    GraphBuilder.to_json(output_path, export_config)
    logger.info(f'Model graphs compared successfully, the result file is saved in {output_path}')


def _build_graph(dump_path, out_path):
    logger.info('Start building model graph...')
    construct_path = FileChecker(os.path.join(dump_path, GraphConst.CONSTRUCT_FILE), FileCheckConst.FILE,
                                 FileCheckConst.READ_ABLE).common_check()
    data_path = FileChecker(os.path.join(dump_path, GraphConst.DUMP_FILE), FileCheckConst.FILE,
                            FileCheckConst.READ_ABLE).common_check()
    create_directory(out_path)
    output_path = os.path.join(out_path, f'build_{current_time}.vis')
    graph = GraphBuilder.build(construct_path, data_path)
    micro_steps = graph.paging_by_micro_step()
    GraphBuilder.to_json(output_path, GraphExportConfig(graph, micro_steps=micro_steps))
    logger.info(f'Model graph built successfully, the result file is saved in {output_path}')


def _graph_service_parser(parser):
    parser.add_argument("-i", "--input_path", dest="input_path", type=str,
                        help="<Required> The compare input path, a dict json.", required=True)
    parser.add_argument("-o", "--output_path", dest="output_path", type=str,
                        help="<Required> The compare task result out path.", required=True)
    parser.add_argument("-dm", "--data_mapping", dest="data_mapping", type=str,
                        help="<optional> The data mapping file path.", required=False)
    parser.add_argument("-lm", "--layer_mapping", dest="layer_mapping", type=str,
                        help="<optional> The layer mapping file path.", required=False)


def _graph_service_command(args):
    with FileOpen(args.input_path, "r") as file:
        input_param = json.load(file)
    npu_path = input_param.get("npu_path")
    bench_path = input_param.get("bench_path")
    if check_file_type(npu_path) == FileCheckConst.DIR and not bench_path:
        _build_graph(npu_path, args.output_path)
    elif check_file_type(npu_path) == FileCheckConst.DIR and check_file_type(bench_path) == FileCheckConst.DIR:
        if args.data_mapping and args.layer_mapping:
            raise RuntimeError('The command line parameters -dm(--data_mapping) and -lm(--layer_mapping) '
                               'cannot be configured at the same time. Only one of them can be configured.')
        _compare_graph(input_param, args)
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
