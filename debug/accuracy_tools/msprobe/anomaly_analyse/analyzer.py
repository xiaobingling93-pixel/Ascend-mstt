from dataclasses import dataclass
import os
import re
from msprobe.core.common.const import Const, CompareConst, FileCheckConst
from msprobe.core.common.file_utils import (load_json, check_file_or_directory_path, check_path_before_create, FileOpen,
                                            change_mode)
from msprobe.core.common.log import logger


def nan_analyze(input_path, output_path):
    path_list = resolve_path(input_path)
    rank_node_dict = {}
    for path in path_list:
        rank_node_dict[path.rank] = find_first_anomaly(path)
    gen_analyze_info(rank_node_dict, output_path)


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


def find_first_anomaly(path):
    cache = FileCache()
    dump_data = cache.load_json(path.dump_path).get('data')
    if not dump_data:
        return {}
    anomaly_info_dict = {}
    index = 0
    for op_name, op_data in dump_data.items():
        if is_communication_op(op_name) or is_ignore_op(op_name):
            continue
        if is_anomaly(op_data):
            anomaly_info = {'dump_info': op_data, 'path_info': path}
            stack_info = cache.load_json(path.stack_path)
            anomaly_info['stack_info'] = find_stack(stack_info, op_name)
            construct_info = cache.load_json(path.construct_path)
            anomaly_info['construct_info'] = find_complete_construct(construct_info, op_name)
            anomaly_info_dict[index] = anomaly_info
        index += 1
    return anomaly_info_dict


def find_stack(stack_info, op_name):
    for item in stack_info:
        if op_name in item[0]:
            return item[1]


def find_complete_construct(construct_info, op_name):
    construct = [op_name]
    while 1:
        op_name = construct_info.get(op_name)
        if not op_name:
            return construct
        construct.insert(0, op_name)


def is_communication_op(op_name):
    # 定义通信算子的关键字，覆盖各种通信操作，如all_reduce, send, broadcast等
    # 从wrap文件中读取，先硬编码在文件中
    communication_keywords = [
        'send',  # send 算子
        'recv',  # recv 算子
        'broadcast',  # broadcast 算子
        'all_reduce',  # all_reduce 算子
        'reduce',  # reduce 算子
        'all_gather',  # all_gather 算子
        'gather',  # gather 算子
        'isend',  # isend 算子
        'irecv',  # irecv 算子
        'scatter',  # scatter 算子
        'reduce_scatter',  # reduce_scatter 算子
        '_reduce_scatter_base',  # _reduce_scatter_base 算子
        '_all_gather_base',  # _all_gather_base 算子
        'all_to_all_single',  # all_to_all_single 算子
        'all_to_all',  # all_to_all 算子
        'all_gather_into_tensor',  # all_gather_into_tensor 算子
        'reduce_scatter_tensor'  # reduce_scatter_tensor 算子
    ]
    return op_name.startswith('Distributed.') and any(keyword in op_name for keyword in communication_keywords)


def is_ignore_op(op_name):
    ignore_keywords = [
        'Torch.empty'
    ]
    return any(keyword in op_name for keyword in ignore_keywords)


def is_anomaly(op_data):
    input_args = op_data.get(Const.INPUT_ARGS, [])
    input_kwargs = op_data.get(Const.INPUT_KWARGS, {})
    input_data = op_data.get(Const.INPUT, [])
    output = op_data.get(Const.OUTPUT, [])
    return any(not has_anomaly(param) for param in [input_args, input_kwargs, input_data]) and has_anomaly(output)


def has_anomaly(param):
    def has_nan_inf(dict_obj, key):
        return str(dict_obj.get(key)).lower() in CompareConst.OVERFLOW_LIST

    items = []
    if isinstance(param, list):
        items = param
    elif isinstance(param, dict):
        items = param.values()
    for item in items:
        if not isinstance(item, dict):
            continue
        if has_nan_inf(item, 'Max') or has_nan_inf(item, 'Min'):
            return True
    return False


def gen_analyze_info(rank_node_dict, output_path):
    check_path_before_create(output_path)
    file_name = 'anomaly_analyze.txt'
    result_file = os.path.join(output_path, file_name)
    try:
        with FileOpen(result_file, 'w+') as output_file:
            output_file.writelines(gen_analyze_content(rank_node_dict))
        change_mode(result_file, FileCheckConst.DATA_FILE_AUTHORITY)
    except IOError as io_error:
        logger.error("Failed to save %s, the reason is %s." % (result_file, io_error))
    else:
        logger.info("The analyze result is saved in: %s" % result_file)


def gen_analyze_content(rank_node_dict):
    result = []
    for i in sorted(list(rank_node_dict.keys())):
        rank_node_dict[i]
    return result


@dataclass
class RankPath:
    rank: int
    dump_path: str
    construct_path: str
    stack_path: str

    def __init__(self, rank, dump_path, construct_path, stack_path):
        self.rank = rank
        check_file_or_directory_path(dump_path)
        self.dump_path = dump_path
        check_file_or_directory_path(construct_path)
        self.construct_path = construct_path
        check_file_or_directory_path(stack_path)
        self.stack_path = stack_path


class FileCache:
    """
    lazy load file
    """
    def __init__(self):
        self._buffer = {}

    def load_json(self, json_path):
        if json_path in self._buffer:
            return self._buffer.get(json_path)
        content = load_json(json_path)
        self._buffer[json_path] = content
        return content
