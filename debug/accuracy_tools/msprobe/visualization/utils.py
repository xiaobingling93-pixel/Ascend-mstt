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
import re
import json
from msprobe.core.common.file_utils import FileOpen
from msprobe.core.common.const import CompareConst, Const
from msprobe.core.compare.acc_compare import Comparator, ModeConfig


def load_json_file(file_path):
    """
    加载json文件
    """
    try:
        with FileOpen(file_path, 'r') as f:
            file_dict = json.load(f)
            if not isinstance(file_dict, dict):
                return {}
            return file_dict
    except json.JSONDecodeError:
        return {}


def load_data_json_file(file_path):
    """
    加载dump.json中的data字段
    """
    return load_json_file(file_path).get(GraphConst.DATA_KEY, {})


def save_json_file(file_path, data):
    """
    保存json文件
    """
    with FileOpen(file_path, 'w') as f:
        f.write(json.dumps(data, indent=4))


def get_csv_df(stack_mode, csv_data, compare_mode):
    """
    调用acc接口写入csv
    """
    dump_mode = GraphConst.GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING.get(compare_mode)
    mode_config = ModeConfig(stack_mode=stack_mode, dump_mode=dump_mode)
    return Comparator(mode_config).make_result_table(csv_data)


def str2float(percentage_str):
    """
    百分比字符串转换转换为浮点型
    Args:
        percentage_str: '0.00%', '23.4%'
    Returns: float 0.00, 0.234
    """
    try:
        percentage_str = percentage_str.strip('%')
        return float(percentage_str) / 100
    except (ValueError, AttributeError):
        return 0


def is_integer(s):
    try:
        int(s)
        return True
    except Exception:
        return False


def check_directory_content(input_path):
    """
    检查input_path内容, 是否全是step{数字}命名的文件夹(例如step0), 或者全是rank{数字}命名的文件夹(例如rank0), 或者全是文件
    """
    contents = os.listdir(input_path)
    if not contents:
        raise ValueError(f'The path {input_path} is empty.')

    # 真实数据dump会有dump_tensor_data文件夹
    if os.path.exists(os.path.join(input_path, Const.DUMP_TENSOR_DATA)):
        return GraphConst.FILES

    # 检查是否全是文件
    if all(os.path.isfile(os.path.join(input_path, item)) for item in contents):
        return GraphConst.FILES

    # 单卡只有一个rank文件夹
    if contents == [Const.RANK]:
        return GraphConst.RANKS

    rank_pattern = re.compile(r'^rank\d+$')
    step_pattern = re.compile(r'^step\d+$')

    rank_all = True
    step_all = True

    for item in contents:
        item_path = os.path.join(input_path, item)
        if not os.path.isdir(item_path):
            continue
        if not rank_pattern.match(item):
            rank_all = False
        if not step_pattern.match(item):
            step_all = False

    if rank_all:
        return GraphConst.RANKS
    if step_all:
        return GraphConst.STEPS

    raise ValueError("The input path content does not conform to the expected naming convention. "
                     "It is expected to be all step{number} named folders (such as step0), "
                     "all rank{number} named folders (such as rank0), or all files.")


class ToolTip:
    MAX_DIFF = 'NPU与标杆API统计信息比对，最大值的差值'
    MIN_DIFF = 'NPU与标杆API统计信息比对，最小值的差值'
    MEAN_DIFF = 'NPU与标杆API统计信息比对，平均值的差值'
    NORM_DIFF = 'NPU与标杆API统计信息比对，2范数（平方根）的差值'
    MD5 = '数据MD5信息，用于比较两个数据信息是否完全一致'
    ONE_THOUSANDTH_ERR_RATIO = 'Tensor中的元素逐个与对应的标杆数据对比，相对误差小于千分之一的比例占总元素个数的比例，比例越接近1越好'
    FIVE_THOUSANDTHS_ERR_RATIO = 'Tensor中的元素逐个与对应的标杆数据对比，相对误差小于千分之五的比例占总元素个数的比例，比例越接近1越好'
    COSINE = (
        '通过计算两个向量的余弦值来判断其相似度，数值越接近于1说明计算出的两个张量越相似，实际可接受阈值为大于0.99。'
        '在计算中可能会存在nan，主要由于可能会出现其中一个向量为0'
    )
    MAX_ABS_ERR = '当最大绝对误差越接近0表示其计算的误差越小，实际可接受阈值为小于0.001'
    MAX_RELATIVE_ERR = (
        '当最大相对误差越接近0表示其计算的误差越小。'
        '当dump数据中存在0或Nan时，比对结果中最大相对误差则出现inf或Nan的情况，属于正常现象'
    )
    SMALL_VALUE_TIP = '{}, 由于{}小于{}, 建议不参考此相对误差，请参考绝对误差'


class GraphConst:
    CONSTRUCT_FILE = 'construct.json'
    DUMP_FILE = 'dump.json'
    STACK_FILE = 'stack.json'
    GRAPH_FILE = 'graph.vis'
    ERROR_KEY = 'error_key'
    SUMMARY_COMPARE = 0
    MD5_COMPARE = 1
    REAL_DATA_COMPARE = 2
    STRUCTURE_COMPARE = 3
    JSON_NPU_KEY = 'NPU'
    JSON_BENCH_KEY = 'Bench'
    JSON_TIP_KEY = 'ToolTip'
    JSON_ROOT_KEY = 'root'
    JSON_NODE_KEY = 'node'
    JSON_DATA_KEY = 'dump_data_dir'
    JSON_TASK_KEY = 'task'
    DATA_KEY = 'data'
    REAL_DATA_TH = 0.1
    MAX_RELATIVE_ERR_TH = 0.5
    ROUND_TH = 6
    JSON_INDEX_KEY = 'precision_index'
    MATCHED_DISTRIBUTED = 'matched_distributed'
    OVERFLOW_LEVEL = 'overflow_level'
    MAX_INDEX_KEY = 1
    MIN_INDEX_KEY = 0
    SUGGEST_KEY = 'text'
    TAG_NA = 'na'
    OUTPUT_INDEX_TWO = -2
    OUTPUT_INDEX_THREE = -3
    OUTPUT_MIN_LEN = 3
    INPUT = '.input.'
    OUTPUT = '.output.'
    STR_MAX_LEN = 50
    SMALL_VALUE = 1e-3
    MD5_INDEX_LIST = [CompareConst.RESULT]
    REAL_DATA_INDEX_LIST = [CompareConst.COSINE, CompareConst.MAX_ABS_ERR, CompareConst.MAX_RELATIVE_ERR,
                            CompareConst.ONE_THOUSANDTH_ERR_RATIO, CompareConst.FIVE_THOUSANDTHS_ERR_RATIO]
    SUMMARY_INDEX_LIST = [CompareConst.MAX_DIFF, CompareConst.MIN_DIFF, CompareConst.MEAN_DIFF,
                          CompareConst.NORM_DIFF, CompareConst.MAX_RELATIVE_ERR, CompareConst.MIN_RELATIVE_ERR,
                          CompareConst.MEAN_RELATIVE_ERR, CompareConst.NORM_RELATIVE_ERR]
    VALUE_INDEX_LIST = [Const.MAX, Const.MIN, Const.MEAN, Const.NORM]
    APIS_BETWEEN_MODULES = 'Apis_Between_Modules'
    NULL = 'null'
    NONE = 'None'
    VALUE = 'value'
    BRACE = '{}'
    DESCRIPTION = 'description'
    COLORS = 'Colors'
    MICRO_STEPS = 'MicroSteps'
    OVERFLOW_CHECK = 'OverflowCheck'

    DUMP_MODE_TO_GRAPHCOMPARE_MODE_MAPPING = {
        Const.ALL: REAL_DATA_COMPARE,
        Const.SUMMARY: SUMMARY_COMPARE,
        Const.MD5: MD5_COMPARE,
        Const.STRUCTURE: STRUCTURE_COMPARE
    }

    GRAPHCOMPARE_MODE_TO_DUMP_MODE_TO_MAPPING = {
        REAL_DATA_COMPARE: Const.ALL,
        SUMMARY_COMPARE: Const.SUMMARY,
        MD5_COMPARE: Const.MD5,
        STRUCTURE_COMPARE: Const.STRUCTURE
    }

    RANKS = 'ranks'
    STEPS = 'steps'
    FILES = 'files'

    SRC = 'src'
    DST = 'dst'

    BATCH_P2P = 'batch_isend_irecv'
    OP = 'op'
    PEER = 'peer'
    GROUP_ID = 'group_id'
