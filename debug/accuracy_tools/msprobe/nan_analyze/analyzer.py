# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

from collections import defaultdict
from multiprocessing import Pool
import os
import re

from msprobe.core.common.file_utils import check_file_or_directory_path, save_json, make_dir
from msprobe.core.common.log import logger
from msprobe.nan_analyze.utils import RankPath, FileCache, is_communication_op, is_ignore_op, NanAnalyseConst
from msprobe.nan_analyze.graph import DataNode, CommunicationNode


def nan_analyze(input_path, output_path):
    path_list = resolve_path(input_path)
    anomaly_nodes = pre_analyze(path_list) # 查找所有出现在通信节点前的异常节点
    if anomaly_nodes:
        gen_analyze_info(anomaly_nodes, output_path)
        return
    logger.info('Do not found anomaly before communication. Start analysing communication.')
    with Pool(processes=max(int((os.cpu_count() + 1) // 4), 1)) as pool:
        def err_call(err):
            logger.error(f'Error occurred while analyze ranks\' communication nodes: {err}')
            try:
                pool.close()
            except OSError as e:
                logger.error(f'Error occurred while terminating the pool: {e}')

        rank_nodes_dict = {}
        for path in path_list:
            rank_nodes_dict[path.rank] = pool.apply_async(analyze_communication_nodes,
                                                          args=(path,), error_callback=err_call)
        rank_nodes_dict = {rank: result.get() for rank, result in rank_nodes_dict.items()}
    connect_communication_nodes(rank_nodes_dict)
    pruning(rank_nodes_dict)
    anomaly_nodes = search_first_anomaly(rank_nodes_dict)
    if anomaly_nodes:
        logger.info(f'Nan analyze finished. Anomaly node is {anomaly_nodes[0].op_name}.')
    gen_analyze_info(anomaly_nodes, output_path)


def resolve_path(step_path):
    check_file_or_directory_path(step_path, True)
    contents = os.listdir(step_path)
    rank_pattern = r'^rank_?(\d+)$'
    dump_path_list = []
    for path in contents:
        match = re.search(rank_pattern, path)
        if not match:
            continue
        rank = int(match.group(1))
        dump_path = os.path.join(step_path, path, NanAnalyseConst.DUMP_FILE)
        construct_path = os.path.join(step_path, path, NanAnalyseConst.CONSTRUCT_FILE)
        stack_path = os.path.join(step_path, path, NanAnalyseConst.STACK_FILE)
        dump_path_list.append(RankPath(rank, dump_path, construct_path, stack_path))
    return dump_path_list


def pre_analyze(path_list):
    logger.info('Start searching anomaly node before communication.')
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
    if not anomaly_nodes:
        logger.info('Cannot find any anomaly node, no need to generate analyze file.')
        return 
    if not os.path.exists(output_path):
        make_dir(output_path)
    file_name = 'anomaly_analyze.json'
    result_file = os.path.join(output_path, file_name)
    result_content = defaultdict(list)
    for node in anomaly_nodes:
        result_content[f'rank_{node.rank}'].append(node.gen_node_info())
    index = 0
    while os.path.exists(result_file):
        result_file = os.path.join(output_path, f'anomaly_analyze_{index}.json')
        index += 1
    save_json(result_file, result_content, 2)
    logger.info(f"The analyze result is saved in: {result_file}")


def analyze_communication_nodes(path: RankPath):
    cache = FileCache()
    data = cache.load_json(path.dump_path).get('data')
    communication_nodes = {}
    skip_flag = True
    compute_ops = []
    last_node_id = None
    sub_layer = 0  # 当前通信算子中异常计算节点的调用序数
    for op_name, op_data in data.items():
        node_id = f'{path.rank}.{op_name}'
        if skip_flag:
            if is_communication_op(op_name):
                skip_flag = False
                communication_nodes[node_id] = CommunicationNode(node_id, path.rank, DataNode(op_name, path, op_data))
                last_node_id = node_id
            continue
        if is_communication_op(op_name):
            comm_node = CommunicationNode(node_id, path.rank, DataNode(op_name, path, op_data))
            comm_node.compute_ops = compute_ops
            compute_ops = []
            last_commu_node = communication_nodes.get(last_node_id)
            if last_commu_node:
                last_commu_node.add_next(comm_node)
            communication_nodes[node_id] = comm_node
            sub_layer = 0
        elif not is_ignore_op(op_name):
            data_node = DataNode(op_name, path, op_data)
            if data_node.is_anomaly():
                data_node.sub_layer = sub_layer
                compute_ops.append(data_node)
                sub_layer += 1
        last_node_id = node_id
    return communication_nodes


def connect_communication_nodes(rank_nodes_dict):
    searched_ranks = set()
    for rank, nodes in rank_nodes_dict.items():
        searched_ranks.add(rank)
        for node in nodes.values():
            connected_nodes = node.find_connected_nodes()
            if not connected_nodes.get('ranks'):
                connected_nodes['ranks'] = rank_nodes_dict.keys()
            for connected_rank in connected_nodes['ranks']:
                if connected_rank in searched_ranks:
                    continue
                tar_node_id = f'{connected_rank}.{connected_nodes["api"]}'
                connected_node = None
                for node_id, _node in rank_nodes_dict[connected_rank].items():
                    if (node_id.startswith(tar_node_id) and _node.type == connected_nodes.get('type') and
                        rank in _node.find_connected_nodes().get('ranks')):
                        connected_node = _node
                        break
                if not connected_node:
                    logger.warning(f'Cannot find connected communication node for "{node.node_id}".')
                    continue
                if connected_node.type == NanAnalyseConst.DST:
                    node.add_dst(connected_node)
                elif connected_node.type == NanAnalyseConst.SRC:
                    connected_node.layer = node.layer
                    connected_node.add_dst(node)
                else:
                    node.add_link(connected_node)


def pruning(rank_nodes_dict):
    deleted_node_id = []
    for nodes in rank_nodes_dict.values():
        for node_id in list(nodes.keys()):
            node = nodes[node_id]
            if not (node.has_nan_inf() or node.compute_ops):
                deleted_node_id.append(node_id)
                node.delete()
                del nodes[node_id]
    logger.debug(f'After pruning, following nodes are removed: [{", ".join(deleted_node_id)}]')


def search_first_anomaly(rank_nodes_dict) -> list:
    # group
    nodes_lists = [sorted(list(nodes.values()), key=lambda x: x.layer) for nodes in rank_nodes_dict.values()]
    forwards = True
    while forwards:
        groups = {}
        group_keys = []
        forwards = False
        for nodes in nodes_lists:
            if nodes:
                node = nodes.pop(0)
                forwards = True
                node_ids_in_same_group = ({node.node_id} | node.link_nodes.keys() | node.src_nodes.keys()
                                          | node.dst_nodes.keys())
                if node_ids_in_same_group in group_keys:
                    groups.get(group_keys.index(node_ids_in_same_group)).append(node)
                else:
                    group_keys.append(node_ids_in_same_group)
                    groups[group_keys.index(node_ids_in_same_group)] = [node]
        anomaly_nodes = []
        for nodes_group in groups.values():
            anomaly_nodes.extend(analyze_anomaly_in_group(nodes_group))
        if anomaly_nodes:
            return [min(anomaly_nodes, key=lambda x: (x.layer, x.sub_layer))]
    return []


def analyze_anomaly_in_group(nodes_group):
    def get_compute_ops_from_commu_nodes(commu_nodes):
        for commu_node in commu_nodes:
            for op_node in commu_node.compute_ops:
                op_node.layer = commu_node.layer
                anomaly_nodes.append(op_node)
    
    def get_commu_ops(commu_nodes):
        for node in commu_nodes:
            node.data.layer = node.layer
            anomaly_nodes.append(node.data)
    
    anomaly_nodes = []
    if all([node.type == NanAnalyseConst.LINK for node in nodes_group]):
        # 筛选入参有问题的通信节点并追溯包含的异常计算节点
        input_anomaly_nodes = list(filter(lambda node: node.input_has_nan_inf(), nodes_group))
        get_compute_ops_from_commu_nodes(input_anomaly_nodes)
        if anomaly_nodes:
            return anomaly_nodes
        # 筛选入参没问题但出参有问题的通信节点
        output_anomaly_nodes = list(filter(lambda node: node.data.is_anomaly(), nodes_group))
        get_commu_ops(output_anomaly_nodes)
        return anomaly_nodes
    else:
        # 先看src中input是否有异常
        src_list = list(filter(lambda node: node.type == NanAnalyseConst.SRC, nodes_group))
        input_anomaly_nodes = list(filter(lambda node: node.input_has_nan_inf(), src_list))
        # 如果有异常回溯计算节点找到异常来源
        get_compute_ops_from_commu_nodes(input_anomaly_nodes)
        if anomaly_nodes:
            return anomaly_nodes
        # 使用cpu模拟节点进行计算，查看结果是否有问题。需要对所有计算节点录入/映射，暂不实现。
        # 筛选dst中output有异常的点
        dst_list = list(filter(lambda node: node.type == NanAnalyseConst.DST and node.data.is_anomaly(), nodes_group))
        get_commu_ops(dst_list)
        return anomaly_nodes


def _nan_analyze_parser(parser):
    parser.add_argument("-i", "--input_path", dest="input_path", default="", type=str,
                        help="<Required> The dump file path, over step level. eg: \"xxx/step_0/\".",
                        required=True)
    parser.add_argument("-o", "--output_path", dest="output_path", default="./output", type=str,
                        help="<optional> The nan inf analyze result output file path.",
                        required=False)


def _run_nan_analyze(args):
    nan_analyze(args.input_path, args.output_path)