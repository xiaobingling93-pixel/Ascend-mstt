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
import re
from msprobe.core.compare.acc_compare import read_op, merge_tensor, get_accuracy
from msprobe.core.common.utils import get_dump_mode
from msprobe.core.common.const import Const
from msprobe.visualization.utils import GraphConst, process_kwargs_parameter
from msprobe.pytorch.compare.pt_compare import PTComparator


# 用于将节点名字解析成对应的NodeOp的规则
op_patterns = [
    r'^(Module)', #NodeOp.module
    r'^(Tensor|Torch|Functional|NPU|VF|Distributed|Aten)' #NodeOp.function_api
]


def get_compare_mode(dump_path_param):
    """
    获得比较模式，包括summary、MD5和真实数据三种模式
    Args:
        dump_path_param: 调用acc_compare接口所依赖的参数
    Returns: 0 summary mode, 1 md5 mode, 2 true data mode
    """
    dump_mode = get_dump_mode(dump_path_param)
    compare_mode = GraphConst.DUMP_MODE_TO_GRAPHCOMPARE_MODE_MAPPING.get(dump_mode)
    return compare_mode


def run_real_data(dump_path_param, csv_path):
    """
    多进程运行生成真实数据
    Args:
        dump_path_param: 调用acc_compare接口所依赖的参数
        csv_path: 生成文件路径
    """
    return PTComparator()._do_multi_process(dump_path_param, csv_path)


def get_input_output(node_data, node_id):
    """
    将dump的原始数据进行拆解，分解为output和input两个数据
    Args:
        node_data: 属于单个节点的dump数据
        node_id: 节点名字
    """
    input_data = {}
    output_data = {}
    op_parsed_list = read_op(node_data, node_id)
    for item in op_parsed_list:
        full_op_name = item.get('full_op_name', '')
        if not full_op_name:
            continue
        splits = full_op_name.split('.')
        if len(splits) < GraphConst.OUTPUT_MIN_LEN:
            continue
        if GraphConst.OUTPUT in splits[GraphConst.OUTPUT_INDEX_TWO] and \
                GraphConst.INPUT not in splits[GraphConst.OUTPUT_INDEX_THREE]:
            output_data[full_op_name] = item
        else:
            input_data[process_kwargs_parameter(full_op_name)] = item
    return input_data, output_data


def compare_data(data_dict_list1, data_dict_list2):
    """
    比较get_input_output中输出的结果是否结构一致，比较一致返回True
    """
    if len(data_dict_list1) != len(data_dict_list2):
        return False
    # 用于比较两个节点是否相等的关键字段
    tag_keys = ['type', 'shape']
    for key1, key2 in zip(data_dict_list1, data_dict_list2):
        dict1 = data_dict_list1[key1]
        dict2 = data_dict_list2[key2]
        for tag_key in tag_keys:
            tag_value1 = dict1.get(tag_key, None)
            tag_value2 = dict2.get(tag_key, None)
            if tag_value1 != tag_value2:
                return False
    return True


def compare_mapping_data(data_dict_list1, data_dict_list2):
    """
    node1映射node2，可能node1参数多于或少于node2参数，个别参数的shape的维度顺序不同，node1参数null对应node2参数其他值
    工具要尽可能保证node的数据能够比对，进行数据的弱校验，仅校验参数的shape维度数值是否相同
    """
    for x, y in zip(data_dict_list1.values(), data_dict_list2.values()):
        x_shape = x.get('shape')
        y_shape = y.get('shape')
        if x_shape is None or y_shape is None:
            continue
        x_shape = sorted(x_shape) if isinstance(x_shape, list) else x_shape
        y_shape = sorted(y_shape) if isinstance(y_shape, list) else y_shape
        if x_shape != y_shape:
            return False
    return True


def format_node_data(data_dict):
    """
    批量进行节点数据的输出
    """
    del_list = ['requires_grad', 'data_name', 'full_op_name']
    for _, value in data_dict.items():
        if not isinstance(value, dict):
            continue
        for item in del_list:
            if item in value:
                del value[item]
        _format_data(value)
    return data_dict


def compare_node(node_ids, data_dicts, stack_json_data, compare_mode):
    """
    调用acc_compare.py中的get_accuracy获得精度对比指标
    真实数据对比模式无法获得精度对比指标，需要调用多进程比对接口
    Returns: 包含参数信息和对比指标（真实数据对比模式除外）的list
    """
    merge_n = _parse_node(node_ids[0], data_dicts[0], stack_json_data, compare_mode)
    merge_b = _parse_node(node_ids[1], data_dicts[1], stack_json_data, compare_mode)
    result = []
    dump_mode = GraphConst.GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING.get(compare_mode)
    get_accuracy(result, merge_n, merge_b, dump_mode)
    return result


def _parse_node(node_id, data_dict, stack_json_data, compare_mode):
    """
    转换节点，使其能够作为acc_compare.py中的get_accuracy的入参
    """
    dump_mode = GraphConst.GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING.get(compare_mode)
    op_parsed_list = read_op(data_dict.get(node_id, {}), node_id)
    if node_id in stack_json_data:
        op_parsed_list.append(
            {'full_op_name': node_id, 'full_info': stack_json_data[node_id]})
    else:
        op_parsed_list.append({'full_op_name': node_id, 'full_info': None})
    result = merge_tensor(op_parsed_list, dump_mode)
    if not result:
        result['op_name'] = []
    return result


def _format_decimal_string(s):
    """
    使用正则表达式匹配包含数字、小数点和可选的百分号的字符串
    """
    pattern = re.compile(r'\d{1,20}\.\d{1,20}%?')
    matches = pattern.findall(s)
    for match in matches:
        is_percent = match.endswith('%')
        number_str = match.rstrip('%')
        decimal_part = number_str.split('.')[1]
        # 如果小数位数大于6，进行处理
        if len(decimal_part) > GraphConst.ROUND_TH:
            number_float = float(number_str)
            formatted_number = f"{number_float:.{GraphConst.ROUND_TH}f}"
            # 如果原来是百分数，加回百分号
            if is_percent:
                formatted_number += '%'
            # 替换原字符串中的数值部分
            s = s.replace(match, formatted_number)
    return s


def _format_data(data_dict):
    """
    格式化数据，小数保留6位，处理一些异常值
    """
    pattern = r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)$'
    none_num = 0
    for key, value in data_dict.items():
        if isinstance(value, str):
            # 将单引号删掉，None换成null避免前端解析错误
            value = value.replace("'", "").replace(GraphConst.NONE, GraphConst.NULL)
            value = _format_decimal_string(value)
        elif value is None or value == ' ':
            value = GraphConst.NULL
        # 科学计数法1.123123123123e-11，格式化为1.123123e-11
        elif isinstance(value, float) and len(str(value)) < GraphConst.STR_MAX_LEN and re.match(pattern, str(value)):
            value = "{:.6e}".format(value)
        elif isinstance(value, float):
            value = round(value, GraphConst.ROUND_TH)
        # Inf会走入这里，确保转成Inf。另外给其他不符合预期的类型做兜底方案
        if not isinstance(value, (list, tuple, dict, str)):
            value = str(value)
        if value == GraphConst.NULL or key == GraphConst.ERROR_KEY:
            none_num += 1
        data_dict[key] = value
    # 字典里的value全null，只保留一个null
    if none_num == len(data_dict):
        data_dict.clear()
        data_dict[GraphConst.VALUE] = GraphConst.NULL
