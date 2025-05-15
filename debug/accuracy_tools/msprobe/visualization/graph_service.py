# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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
from copy import deepcopy
from multiprocessing import cpu_count, Pool
from msprobe.core.common.file_utils import (check_file_type, create_directory, FileChecker,
                                            check_file_or_directory_path, load_json)
from msprobe.core.common.const import FileCheckConst, Const
from msprobe.core.common.utils import CompareException, get_dump_mode
from msprobe.visualization.compare.graph_comparator import GraphComparator
from msprobe.visualization.utils import GraphConst, check_directory_content, SerializableArgs
from msprobe.visualization.builder.graph_builder import GraphBuilder, GraphExportConfig, GraphInfo, BuildGraphTaskInfo
from msprobe.core.common.log import logger
from msprobe.visualization.graph.node_colors import NodeColors
from msprobe.core.compare.layer_mapping import generate_api_mapping_by_layer_mapping
from msprobe.core.compare.utils import check_and_return_dir_contents
from msprobe.core.common.utils import detect_framework_by_dump_json
from msprobe.visualization.graph.distributed_analyzer import DistributedAnalyzer

current_time = time.strftime("%Y%m%d%H%M%S")


def _compare_graph(graph_n: GraphInfo, graph_b: GraphInfo, input_param, args):
    dump_path_param = {
        'npu_json_path': graph_n.data_path,
        'bench_json_path': graph_b.data_path,
        'stack_json_path': graph_n.stack_path,
        'is_print_compare_log': input_param.get("is_print_compare_log", True)
    }
    mapping_dict = {}
    if args.layer_mapping:
        try:
            mapping_dict = generate_api_mapping_by_layer_mapping(graph_n.data_path, graph_b.data_path,
                                                                 args.layer_mapping)
        except Exception:
            logger.warning('The layer mapping file parsing failed, please check file format, mapping is not effective.')
    is_cross_framework = detect_framework_by_dump_json(graph_n.data_path) != \
                         detect_framework_by_dump_json(graph_b.data_path)
    if is_cross_framework and not args.layer_mapping:
        logger.error('The cross_frame graph comparison failed. '
                     'Please specify -lm or --layer_mapping when performing cross_frame graph comparison.')
        raise CompareException(CompareException.CROSS_FRAME_ERROR)

    graph_comparator = GraphComparator([graph_n.graph, graph_b.graph], dump_path_param, args, is_cross_framework,
                                       mapping_dict=mapping_dict)
    graph_comparator.compare()
    return graph_comparator


def _compare_graph_result(input_param, args):
    logger.info('Start building model graphs...')
    # 对两个数据进行构图
    graph_n = _build_graph_info(input_param.get('npu_path'), args)
    graph_b = _build_graph_info(input_param.get('bench_path'), args)
    logger.info('Model graphs built successfully, start Comparing graphs...')
    # 基于graph、stack和data进行比较
    graph_comparator = _compare_graph(graph_n, graph_b, input_param, args)
    # 增加micro step标记
    micro_steps = graph_n.graph.paging_by_micro_step(graph_b.graph)
    # 开启溢出检测
    if args.overflow_check:
        graph_n.graph.overflow_check()
        graph_b.graph.overflow_check()

    return CompareGraphResult(graph_n.graph, graph_b.graph, graph_comparator, micro_steps)


def _export_compare_graph_result(args, result):
    graphs = [result.graph_n, result.graph_b]
    graph_comparator = result.graph_comparator
    micro_steps = result.micro_steps
    output_file_name = result.output_file_name
    if not output_file_name:
        output_file_name = f'compare_{current_time}.vis'
    logger.info(f'Start exporting compare graph result, file name: {output_file_name}...')
    output_path = os.path.join(args.output_path, output_file_name)
    task = GraphConst.GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING.get(graph_comparator.ma.compare_mode)
    export_config = GraphExportConfig(graphs[0], graphs[1], graph_comparator.ma.get_tool_tip(),
                                      NodeColors.get_node_colors(graph_comparator.ma.compare_mode), micro_steps, task,
                                      args.overflow_check, graph_comparator.ma.compare_mode)
    try:
        GraphBuilder.to_json(output_path, export_config)
        logger.info(f'Exporting compare graph result successfully, the result file is saved in {output_path}')
        return ''
    except RuntimeError as e:
        logger.error(f'Failed to export compare graph result, file: {output_file_name}, error: {e}')
        return output_file_name


def _build_graph_info(dump_path, args):
    construct_path = FileChecker(os.path.join(dump_path, GraphConst.CONSTRUCT_FILE), FileCheckConst.FILE,
                                 FileCheckConst.READ_ABLE).common_check()
    data_path = FileChecker(os.path.join(dump_path, GraphConst.DUMP_FILE), FileCheckConst.FILE,
                            FileCheckConst.READ_ABLE).common_check()
    stack_path = FileChecker(os.path.join(dump_path, GraphConst.STACK_FILE), FileCheckConst.FILE,
                             FileCheckConst.READ_ABLE).common_check()
    graph = GraphBuilder.build(construct_path, data_path, stack_path, complete_stack=args.complete_stack)
    return GraphInfo(graph, construct_path, data_path, stack_path)


def _build_graph_result(dump_path, args):
    logger.info('Start building model graphs...')
    graph = _build_graph_info(dump_path, args).graph
    # 增加micro step标记
    micro_steps = graph.paging_by_micro_step()
    # 开启溢出检测
    if args.overflow_check:
        graph.overflow_check()
    return BuildGraphResult(graph, micro_steps)


def _run_build_graph_compare(input_param, args, nr, br):
    logger.info(f'Start building graph for {nr}...')
    graph_n = _build_graph_info(input_param.get('npu_path'), args)
    graph_b = _build_graph_info(input_param.get('bench_path'), args)
    logger.info(f'Building graph for {nr} finished.')
    return BuildGraphTaskInfo(graph_n, graph_b, nr, br, current_time)


def _run_build_graph_single(dump_ranks_path, rank, step, args):
    logger.info(f'Start building graph for {rank}...')
    dump_path = os.path.join(dump_ranks_path, rank)
    output_file_name = f'build_{step}_{rank}_{current_time}.vis' if step else f'build_{rank}_{current_time}.vis'
    result = _build_graph_result(dump_path, args)
    result.output_file_name = output_file_name
    if rank != Const.RANK:
        try:
            result.rank = int(rank.replace(Const.RANK, ""))
        except Exception as e:
            logger.error('The folder name format is incorrect, expected rank+number.')
            raise CompareException(CompareException.INVALID_PATH_ERROR) from e
    logger.info(f'Building graph for step: {step}, rank: {rank} finished.')
    return result


def _run_graph_compare(graph_task_info, input_param, args, output_file_name):
    logger.info(f'Start comparing data for {graph_task_info.npu_rank}...')
    graph_n = graph_task_info.graph_info_n
    graph_b = graph_task_info.graph_info_b
    nr = graph_task_info.npu_rank
    graph_comparator = _compare_graph(graph_n, graph_b, input_param, args)
    micro_steps = graph_n.graph.paging_by_micro_step(graph_b.graph)
    # 开启溢出检测
    if args.overflow_check:
        graph_n.graph.overflow_check()
        graph_b.graph.overflow_check()
    graph_result = CompareGraphResult(graph_n.graph, graph_b.graph, graph_comparator, micro_steps)
    graph_result.output_file_name = output_file_name
    if nr != Const.RANK:
        try:
            graph_result.rank = int(nr.replace(Const.RANK, ""))
        except Exception as e:
            logger.error('The folder name format is incorrect, expected rank+number.')
            raise CompareException(CompareException.INVALID_PATH_ERROR) from e
    logger.info(f'Comparing data for {graph_task_info.npu_rank} finished.')
    return graph_result


def _export_build_graph_result(args, result):
    out_path = args.output_path
    graph = result.graph
    micro_steps = result.micro_steps
    overflow_check = args.overflow_check
    output_file_name = result.output_file_name
    if not output_file_name:
        output_file_name = f'build_{current_time}.vis'
    logger.info(f'Start exporting graph for {output_file_name}...')
    output_path = os.path.join(out_path, output_file_name)
    try:
        GraphBuilder.to_json(output_path, GraphExportConfig(graph, micro_steps=micro_steps,
                                                            overflow_check=overflow_check))
        logger.info(f'Model graph exported successfully, the result file is saved in {output_path}')
        return None
    except RuntimeError as e:
        logger.error(f'Failed to export model graph, file: {output_file_name}, error: {e}')
        return output_file_name


def is_real_data_compare(input_param, npu_ranks, bench_ranks):
    dump_rank_n = input_param.get('npu_path')
    dump_rank_b = input_param.get('bench_path')
    has_real_data = False
    for nr, br in zip(npu_ranks, bench_ranks):
        dump_path_param = {
            'npu_json_path': FileChecker(os.path.join(dump_rank_n, nr, GraphConst.DUMP_FILE), FileCheckConst.FILE,
                                         FileCheckConst.READ_ABLE).common_check(),
            'bench_json_path': FileChecker(os.path.join(dump_rank_b, br, GraphConst.DUMP_FILE), FileCheckConst.FILE,
                                           FileCheckConst.READ_ABLE).common_check()
        }
        has_real_data |= get_dump_mode(dump_path_param) == Const.ALL
    return has_real_data


def _mp_compare(input_param, serializable_args, output_file_name, nr, br):
    graph_task_info = _run_build_graph_compare(input_param, serializable_args, nr, br)
    return _run_graph_compare(graph_task_info, input_param, serializable_args, output_file_name)


def _compare_graph_ranks(input_param, args, step=None):
    with Pool(processes=max(int((cpu_count() + 1) // 4), 1)) as pool:
        def err_call(err):
            logger.error(f'Error occurred while comparing graph ranks: {err}')
            try:
                pool.close()
            except OSError as e:
                logger.error(f'Error occurred while terminating the pool: {e}')

        serializable_args = SerializableArgs(args)
        # 暂存所有rank的graph，用于匹配rank间的分布式节点
        compare_graph_results = _get_compare_graph_results(input_param, serializable_args, step, pool, err_call)

        # 匹配rank间的分布式节点
        if len(compare_graph_results) > 1:
            DistributedAnalyzer({obj.rank: obj.graph_n for obj in compare_graph_results},
                                args.overflow_check).distributed_match()
            DistributedAnalyzer({obj.rank: obj.graph_b for obj in compare_graph_results},
                                args.overflow_check).distributed_match()

        export_res_task_list = []
        create_directory(args.output_path)
        for result in compare_graph_results:
            export_res_task_list.append(pool.apply_async(_export_compare_graph_result,
                                                         args=(serializable_args, result),
                                                         error_callback=err_call))
        export_res_list = [res.get() for res in export_res_task_list]
        if any(export_res_list):
            failed_names = list(filter(lambda x: x, export_res_list))
            logger.error(f'Unable to export compare graph results: {", ".join(failed_names)}.')
        else:
            logger.info('Successfully exported compare graph results.')


def _get_compare_graph_results(input_param, serializable_args, step, pool, err_call):
    dump_rank_n = input_param.get('npu_path')
    dump_rank_b = input_param.get('bench_path')
    npu_ranks = sorted(check_and_return_dir_contents(dump_rank_n, Const.RANK))
    bench_ranks = sorted(check_and_return_dir_contents(dump_rank_b, Const.RANK))
    if npu_ranks != bench_ranks:
        logger.error('The number of ranks in the two runs are different. Unable to match the ranks.')
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    compare_graph_results = []
    if is_real_data_compare(input_param, npu_ranks, bench_ranks):
        mp_task_dict = {}
        for nr, br in zip(npu_ranks, bench_ranks):
            input_param['npu_path'] = os.path.join(dump_rank_n, nr)
            input_param['bench_path'] = os.path.join(dump_rank_b, br)
            output_file_name = f'compare_{step}_{nr}_{current_time}.vis' if step else f'compare_{nr}_{current_time}.vis'
            input_param_copy = deepcopy(input_param)
            mp_task_dict[output_file_name] = pool.apply_async(_run_build_graph_compare,
                                                              args=(input_param_copy, serializable_args, nr, br),
                                                              error_callback=err_call)

        mp_res_dict = {k: v.get() for k, v in mp_task_dict.items()}
        for output_file_name, mp_res in mp_res_dict.items():
            compare_graph_results.append(_run_graph_compare(mp_res, input_param, serializable_args, output_file_name))
    else:
        compare_graph_tasks = []
        for nr, br in zip(npu_ranks, bench_ranks):
            input_param['npu_path'] = os.path.join(dump_rank_n, nr)
            input_param['bench_path'] = os.path.join(dump_rank_b, br)
            output_file_name = f'compare_{step}_{nr}_{current_time}.vis' if step else f'compare_{nr}_{current_time}.vis'
            input_param_copy = deepcopy(input_param)
            compare_graph_tasks.append(pool.apply_async(_mp_compare,
                                                        args=(input_param_copy, serializable_args, output_file_name, nr,
                                                              br),
                                                        error_callback=err_call))
        compare_graph_results = [task.get() for task in compare_graph_tasks]
    return compare_graph_results


def _compare_graph_steps(input_param, args):
    dump_step_n = input_param.get('npu_path')
    dump_step_b = input_param.get('bench_path')

    npu_steps = sorted(check_and_return_dir_contents(dump_step_n, Const.STEP))
    bench_steps = sorted(check_and_return_dir_contents(dump_step_b, Const.STEP))

    if npu_steps != bench_steps:
        logger.error('The number of steps in the two runs is different. Unable to match the steps.')
        raise CompareException(CompareException.INVALID_PATH_ERROR)

    for folder_step in npu_steps:
        logger.info(f'Start processing data for {folder_step}...')
        input_param['npu_path'] = os.path.join(dump_step_n, folder_step)
        input_param['bench_path'] = os.path.join(dump_step_b, folder_step)

        _compare_graph_ranks(input_param, args, step=folder_step)


def _build_graph_ranks(dump_ranks_path, args, step=None):
    ranks = sorted(check_and_return_dir_contents(dump_ranks_path, Const.RANK))
    serializable_args = SerializableArgs(args)
    with Pool(processes=max(int((cpu_count() + 1) // 4), 1)) as pool:
        def err_call(err):
            logger.error(f'Error occurred while comparing graph ranks: {err}')
            try:
                pool.close()
            except OSError as e:
                logger.error(f'Error occurred while terminating the pool: {e}')

        build_graph_tasks = []
        for rank in ranks:
            build_graph_tasks.append(pool.apply_async(_run_build_graph_single,
                                                      args=(dump_ranks_path, rank, step, serializable_args),
                                                      error_callback=err_call))
        build_graph_results = [task.get() for task in build_graph_tasks]

        if len(build_graph_results) > 1:
            DistributedAnalyzer({obj.rank: obj.graph for obj in build_graph_results},
                                args.overflow_check).distributed_match()

        create_directory(args.output_path)
        export_build_graph_tasks = []
        for result in build_graph_results:
            export_build_graph_tasks.append(pool.apply_async(_export_build_graph_result,
                                                             args=(serializable_args, result),
                                                             error_callback=err_call))
        export_build_graph_result = [task.get() for task in export_build_graph_tasks]
        if any(export_build_graph_result):
            failed_names = list(filter(lambda x: x, export_build_graph_result))
            logger.error(f'Unable to export build graph results: {failed_names}.')
        else:
            logger.info(f'Successfully exported build graph results.')



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
    parser.add_argument("-lm", "--layer_mapping", dest="layer_mapping", type=str, nargs='?', const=True,
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
            result = _build_graph_result(npu_path, args)
            create_directory(args.output_path)
            file_name = _export_build_graph_result(args, result)
            if file_name:
                logger.error('Failed to export model build graph.')
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
            result = _compare_graph_result(input_param, args)
            create_directory(args.output_path)
            file_name = _export_compare_graph_result(args, result)
            if file_name:
                logger.error('Failed to export model compare graph.')
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
