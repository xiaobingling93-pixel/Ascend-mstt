import os
import re
from multiprocessing import Pool
from collections import defaultdict

from msprobe.core.common.const import Const, FileCheckConst
from msprobe.core.common.file_utils import (check_file_or_directory_path, check_path_before_create, FileOpen,
                                            change_mode)
from msprobe.core.common.log import logger

from core.common.file_utils import save_json
# from msprobe.anomaly_analyse.utils import RankPath, FileCache, is_communication_op, is_ignore_op
from debug.accuracy_tools.msprobe.anomaly_analyse.utils import RankPath, FileCache, is_communication_op, is_ignore_op
# from msprobe.anomaly_analyze.graph import DataNode, CommunicationNode
from graph import DataNode, CommunicationNode


def nan_analyze(input_path, output_path):
    path_list = resolve_path(input_path)
    anomaly_nodes = pre_analyze(path_list) # 查找所有出现在通信节点前的异常节点
    if anomaly_nodes:
        gen_analyze_info(anomaly_nodes, output_path)
        return
    with Pool(processes=max(int((os.cpu_count() + 1) // 4), 1)) as pool:
        def err_call(err):
            logger.error(f'Error occurred while analyze ranks\' communication nodes: {err}')
            try:
                pool.close()
            except OSError as e:
                logger.error(f'Error occurred while terminating the pool: {e}')

        rank_nodes_dict = {}
        for path in path_list:
            rank_nodes_dict[path.rank] = pool.appy_async(analyze_communication_nodes,
                                                         args=(path,), error_callback=err_call)
        rank_nodes_dict = {rank: result.get() for rank, result in rank_nodes_dict.items()}
    connect_communication_nodes(rank_nodes_dict)
    color_and_pruning(rank_nodes_dict)
    anomaly_nodes = search_first_anomaly(rank_nodes_dict)
    gen_analyze_info(anomaly_nodes, output_path)


def resolve_path(step_path):
    check_file_or_directory_path(step_path, True)
    contents = os.listdir(step_path)
    rank_pattern = r'^rank_\d+$'
    dump_path_list = []
    for path in contents:
        if not re.match(rank_pattern, path):
            continue
        rank = int(path.split('_')[1])
        dump_path = os.path.join(step_path, path, Const.DUMP_FILE)
        construct_path = os.path.join(step_path, path, Const.CONSTRUCT_FILE)
        stack_path = os.path.join(step_path, path, Const.STACK_FILE)
        dump_path_list.append(RankPath(rank, dump_path, construct_path, stack_path))
    return dump_path_list


def pre_analyze(path_list):
    cache = FileCache()
    anomaly_nodes = []
    for path in path_list:
        dump_data = cache.load_json(path.dump_path).get('data')
        if not dump_data:
            logger.warning(f'Rank {path.rank} has no dump data!')
            continue
        for op_name, op_data in dump_data.items():
            if is_communication_op(op_name):
                break
            data_node = DataNode(op_name, path, op_data)
            if data_node.is_anomaly():
                anomaly_nodes.append(data_node)
    return anomaly_nodes


def gen_analyze_info(anomaly_nodes, output_path):
    check_path_before_create(output_path)
    file_name = 'anomaly_analyze.txt'
    result_file = os.path.join(output_path, file_name)
    result_content = defaultdict(list)
    for node in anomaly_nodes:
        result_content[f'rank_{node.rank}'].append(node.gen_node_info())
    save_json(output_path, result_content, 2)
    logger.info("The analyze result is saved in: %s" % result_file)


def analyze_communication_nodes(path: RankPath):
    cache = FileCache()
    data = cache.load_json(path.dump_path).get('data')
    communication_nodes = []
    skip_flag = True
    compute_ops = []
    for op_name, op_data in data.items():
        if is_communication_op(op_name):
            skip_flag = False
            communication_nodes.append(CommunicationNode(f'{path.rank}.{op_name}', path.rank,
                                                         DataNode(op_name, path, op_data)))
            continue
        if skip_flag:
            continue
        if is_communication_op(op_name):
            comm_node = CommunicationNode(f'{path.rank}.{op_name}', path.rank, op_data)
            comm_node.compute_ops = compute_ops
            compute_ops = []
            communication_nodes[-1].add_next(comm_node)
            communication_nodes.append(comm_node)
        elif not is_ignore_op(op_name):
            data_node = DataNode(op_name, path, op_data)
            if data_node.is_anomaly():
                compute_ops.append(data_node)
    return communication_nodes


def connect_communication_nodes(rank_nodes_dict):
    for rank, nodes in rank_nodes_dict.items():
        for node in nodes:
            connected_nodes = node.find_connected_nodes()

