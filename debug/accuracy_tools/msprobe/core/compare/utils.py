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

import os
import re
import math
import zlib
from dataclasses import dataclass

import numpy as np

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.utils import CompareException, check_regex_prefix_format_valid, logger
from msprobe.core.common.file_utils import check_file_or_directory_path


def extract_json(dirname, stack_json=False):
    json_path = ''
    for fname in os.listdir(dirname):
        if fname == "construct.json":
            continue
        full_path = os.path.join(dirname, fname)
        if full_path.endswith('.json'):
            json_path = full_path
            if not stack_json and 'stack' not in json_path:
                break
            if stack_json and 'stack' in json_path:
                break

    # Provide robustness on invalid directory inputs
    if not json_path:
        logger.error(f'No file is found in dump dir {dirname}. ')
        raise CompareException(CompareException.NO_DUMP_FILE_ERROR)
    return json_path


def check_and_return_dir_contents(dump_dir, prefix):
    """
    check the given dump dir and validate files in dump dir by using the given prefix patterns to build a
    pattern: ^{prefix}(?:0|[0-9][1-9]*)?$

    Args:
        dump_dir (str): dump dir
        prefix (str): prefix for the patterns, prefix should be less than 20 characters and alphanumeric/-/_ only

    Returns:
        content [list]: dir contents
    Raises:
        CompareException: invalid path
        ValueError: prefix not match the patterns

    """
    check_regex_prefix_format_valid(prefix)
    check_file_or_directory_path(dump_dir, True)
    contents = os.listdir(dump_dir)
    pattern = re.compile(rf'^{prefix}(?:0|[0-9][1-9]*)?$')
    for name in contents:
        if not pattern.match(name):
            logger.error(
                f"dump_dir contains '{name}'. Expected '{prefix}'. This name is not in the format of dump "
                f"output. Please check and delete irrelevant files in {dump_dir} and try again."
            )
            raise CompareException(CompareException.INVALID_PATH_ERROR)
    return contents


def rename_api(npu_name, process):
    npu_split = npu_name.split(process)
    try:
        torch_func_index, in_out = npu_split[0], npu_split[1]
    except IndexError as error:
        logger.error(f'{npu_name} can not be split with {process}, please check!')
        raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from error
    torch_func_split = torch_func_index.rsplit(Const.SEP, 2)
    torch_func = str(torch_func_split[0]) + str(in_out)
    return torch_func


def read_op(op_data, op_name):
    io_name_mapping = {
        Const.INPUT_ARGS: '.input',
        Const.INPUT_KWARGS: '.input',
        Const.INPUT: '.input',
        Const.OUTPUT: '.output'
    }

    op_parsed_list = []
    for name in io_name_mapping:
        if name in op_data:
            op_parsed_list.extend(op_item_parse(op_data[name], op_name + io_name_mapping[name]))
    return op_parsed_list


def op_item_parse(op_data, op_name: str) -> list:
    default_item = {
        'full_op_name': op_name,
        'type': None,
        'Max': None,
        'Min': None,
        'Mean': None,
        'Norm': None,
        'dtype': None,
        'shape': None,
        'md5': None,
        'value': None,
        'data_name': '-1'
    }

    if op_data is None:
        return [default_item]
    elif not op_data:
        return []
    
    item_list = []
    if isinstance(op_data, list):
        for i, data in enumerate(op_data):
            item_list.extend(op_item_parse(data, op_name + Const.SEP + str(i)))
    elif isinstance(op_data, dict):
        if is_tensor(op_data):
            return [gen_op_item(op_data, op_name)]
        for sub_name, sub_data in op_data.items():
            item_list.extend(op_item_parse(sub_data, op_name + Const.SEP + str(sub_name)))
    return item_list


def is_tensor(op_data):
    return 'type' in op_data and isinstance(op_data['type'], str)


def gen_op_item(op_data, op_name):
    op_item = {}
    op_item.update(op_data)
    op_item['full_op_name'] = op_name
    op_item['data_name'] = op_data.get('data_name', '-1')
    params = ['Max', 'Min', 'Mean', 'Norm']
    for i in params:
        if i not in op_item:
            op_item[i] = None
    
    if not op_item.get('dtype'):
        if op_item.get('type') == 'torch.Size':
            op_item['dtype'] = op_data.get('type')
            op_item['shape'] = str(op_data.get('value'))
        elif op_item.get('type') == 'slice':
            op_item['dtype'] = op_data.get('type')
            op_item['shape'] = str(np.shape(np.array(op_data.get('value'))))
        else:
            op_item['dtype'] = str(type(op_data.get('value')))
            op_item['shape'] = '[]'
            for i in params:
                op_item[i] = op_data.get('value')
    if not op_item.get('md5'):
        op_item['md5'] = f"{zlib.crc32(str(op_data.get('value', '')).encode()):08x}"
    
    return op_item


def resolve_api_special_parameters(data_dict, full_op_name, item_list):
    """
    Function Description:
        解析下面格式的数据, 是api参数的一种特殊格式
        {
         "last_hidden_state": {
          "type": "torch.Tensor",
          "dtype": "torch.bfloat16",
          ...
         },
         "loss": {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          ...
         }
        }
    Parameter:
        data_dict: 字典格式的数据
        full_op_name: 参数的全名字符串
        item_list: 参数信息集合        
    """
    for key, value in data_dict.items():
        if isinstance(value, dict):
            parsed_item = value
            parts = full_op_name.split(Const.SEP)
            parts.insert(-1, key)
            full_op_name_new = ".".join(parts)
            parsed_item['full_op_name'] = full_op_name_new
            item_list.append(parsed_item)


def process_summary_data(summary_data):
    """处理summary_data中的nan值，返回处理后的列表"""
    return [CompareConst.NAN if isinstance(x, float) and math.isnan(x) else x for x in summary_data]


def get_rela_diff_summary_mode(result_item, npu_summary_data, bench_summary_data, err_msg):
    start_idx = CompareConst.SUMMARY_COMPARE_RESULT_HEADER.index(CompareConst.MAX_DIFF)
    warning_flag = False
    for i, (npu_val, bench_val) in enumerate(zip(npu_summary_data, bench_summary_data)):
        if all(isinstance(val, (float, int)) and not isinstance(val, bool) for val in [npu_val, bench_val]):
            diff = npu_val - bench_val
            if math.isnan(diff):
                diff = CompareConst.NAN
                relative = CompareConst.NAN
            else:
                if bench_val != 0:
                    relative = str(abs((diff / bench_val) * 100)) + '%'
                else:
                    relative = CompareConst.N_A
                magnitude_diff = abs(diff) / (max(abs(npu_val), abs(bench_val)) + CompareConst.EPSILON)
                if magnitude_diff > CompareConst.MAGNITUDE:
                    warning_flag = True
            result_item[start_idx + i] = diff
            result_item[start_idx + i + CompareConst.STATISTICS_INDICATOR_NUM] = relative
        else:
            result_item[start_idx + i] = CompareConst.N_A
            result_item[start_idx + i + CompareConst.STATISTICS_INDICATOR_NUM] = CompareConst.N_A

    accuracy_check = CompareConst.WARNING if warning_flag else ""
    err_msg += "Need double check api accuracy." if warning_flag else ""
    for i in range(start_idx, len(result_item)):
        if str(result_item[i]) in ('inf', '-inf', 'nan'):
            result_item[i] = f'{result_item[i]}\t'
    return result_item, accuracy_check, err_msg


def get_accuracy(result, n_dict, b_dict, dump_mode):
    def get_accuracy_core(n_start, n_len, b_start, b_len, key):
        min_len = min(n_len, b_len)
        npu_stack_info = n_dict.get("stack_info", None)
        bench_stack_info = b_dict.get("stack_info", None)
        has_stack = npu_stack_info and bench_stack_info

        if dump_mode == Const.ALL:
            npu_data_name = n_dict.get("data_name", None)
            bench_data_name = b_dict.get("data_name", None)

        for index in range(min_len):
            n_name = n_dict['op_name'][n_start + index]
            b_name = b_dict['op_name'][b_start + index]
            n_struct = n_dict[key][index]
            b_struct = b_dict[key][index]
            err_msg = ""
            if dump_mode == Const.MD5:
                result_item = [
                    n_name, b_name, n_struct[0], b_struct[0], n_struct[1], b_struct[1], n_struct[2], b_struct[2],
                    CompareConst.PASS if n_struct[2] == b_struct[2] else CompareConst.DIFF
                ]
                if has_stack and index == 0 and key == "input_struct":
                    result_item.extend(npu_stack_info)
                else:
                    result_item.append(CompareConst.NONE)
                result.append(result_item)
                continue

            if dump_mode == Const.SUMMARY:
                result_item = [
                    n_name, b_name, n_struct[0], b_struct[0], n_struct[1], b_struct[1],
                    " ", " ", " ", " ", " ", " ", " ", " "
                ]
            else:
                result_item = [
                    n_name, b_name, n_struct[0], b_struct[0], n_struct[1], b_struct[1],
                    " ", " ", " ", " ", " "
                ]

            npu_summary_data = n_dict.get(CompareConst.SUMMARY)[n_start + index]
            bench_summary_data = b_dict.get(CompareConst.SUMMARY)[b_start + index]
            result_item.extend(process_summary_data(npu_summary_data))
            result_item.extend(process_summary_data(bench_summary_data))

            if dump_mode == Const.SUMMARY:
                result_item, accuracy_check, err_msg = get_rela_diff_summary_mode(result_item, npu_summary_data,
                                                                                  bench_summary_data, err_msg)

            result_item.append(accuracy_check if dump_mode == Const.SUMMARY else CompareConst.ACCURACY_CHECK_YES)
            result_item.append(err_msg)
            if has_stack and index == 0 and key == "input_struct":
                result_item.extend(npu_stack_info)
            else:
                result_item.append(CompareConst.NONE)
            if dump_mode == Const.ALL:
                result_item.append(npu_data_name[n_start + index])

            result.append(result_item)

        if n_len > b_len:
            for index in range(b_len, n_len):
                n_name = n_dict['op_name'][n_start + index]
                n_struct = n_dict[key][index]
                if dump_mode == Const.MD5:
                    result_item = [
                        n_name, CompareConst.NAN, n_struct[0], CompareConst.NAN, n_struct[1], CompareConst.NAN,
                        n_struct[2], CompareConst.NAN, CompareConst.NAN
                    ]
                    result.append(result_item)
                    continue
                result_item = [
                    n_name, CompareConst.NAN, n_struct[0], CompareConst.NAN, n_struct[1], CompareConst.NAN,
                    " ", " ", " ", " ", " "
                ]
                summary_data = n_dict.get(CompareConst.SUMMARY)[n_start + index]
                result_item.extend(summary_data)
                summary_data = [CompareConst.NAN for _ in range(len(n_dict.get(CompareConst.SUMMARY)[0]))]
                result_item.extend(summary_data)

                err_msg = ""
                result_item.append(CompareConst.ACCURACY_CHECK_YES)
                result_item.append(err_msg)

                if has_stack and index == 0 and key == "input_struct":
                    result_item.extend(npu_stack_info)
                else:
                    result_item.append(CompareConst.NONE)
                if dump_mode == Const.ALL:
                    result_item.append(npu_data_name[n_start + index])

                result.append(result_item)

    n_num = len(n_dict['op_name'])
    b_num = len(b_dict['op_name'])
    n_num_input = len([name for name in n_dict['op_name']
                       if Const.INPUT in name.split(Const.SEP) or Const.KWARGS in name.split(Const.SEP)])
    b_num_input = len([name for name in b_dict['op_name']
                       if Const.INPUT in name.split(Const.SEP) or Const.KWARGS in name.split(Const.SEP)])
    n_num_output = n_num - n_num_input
    b_num_output = b_num - b_num_input
    get_accuracy_core(0, n_num_input, 0, b_num_input, 'input_struct')
    get_accuracy_core(n_num_input, n_num_output, b_num_input, b_num_output, 'output_struct')


def get_un_match_accuracy(result, n_dict, dump_mode):
    index_out = 0
    npu_stack_info = n_dict.get("stack_info", None)
    bench_name, bench_type, bench_shape = CompareConst.N_A, CompareConst.N_A, CompareConst.N_A
    err_msg = CompareConst.NO_BENCH
    accuracy_check_res = CompareConst.N_A
    for index, n_name in enumerate(n_dict["op_name"]):
        name_ele_list = n_name.split(Const.SEP)
        if Const.INPUT in name_ele_list or Const.KWARGS in name_ele_list:
            n_struct = n_dict[CompareConst.INPUT_STRUCT][index]
        if Const.OUTPUT in name_ele_list:
            n_struct = n_dict[CompareConst.OUTPUT_STRUCT][index_out]
            index_out += 1

        result_item = [n_name, bench_name, n_struct[0], bench_type, n_struct[1], bench_shape]
        if dump_mode == Const.MD5:
            result_item.extend([CompareConst.N_A] * 3)
            if npu_stack_info and index == 0:
                result_item.extend(npu_stack_info)
            else:
                result_item.append(CompareConst.NONE)
            result.append(result_item)
            continue
        if dump_mode == Const.SUMMARY:
            result_item.extend([CompareConst.N_A] * 8)
        else:
            result_item.extend([CompareConst.N_A] * 5)
        npu_summary_data = n_dict.get("summary")[index]
        result_item.extend(npu_summary_data)
        bench_summary_data = [CompareConst.N_A] * 4
        result_item.extend(bench_summary_data)
        result_item.append(accuracy_check_res)
        result_item.append(err_msg)
        if npu_stack_info and index == 0:
            result_item.extend(npu_stack_info)
        else:
            result_item.append(CompareConst.NONE)
        if dump_mode == Const.ALL and result_item[1] == CompareConst.N_A:
            result_item.extend(["-1"])
        result.append(result_item)


def merge_tensor(tensor_list, dump_mode):
    op_dict = {}
    op_dict["op_name"] = []
    op_dict[CompareConst.INPUT_STRUCT] = []
    op_dict[CompareConst.KWARGS_STRUCT] = []
    op_dict[CompareConst.OUTPUT_STRUCT] = []
    op_dict[Const.SUMMARY] = []
    op_dict["stack_info"] = []

    if dump_mode == Const.ALL:
        op_dict["data_name"] = []

    for tensor in tensor_list:
        if len(tensor) == 2:
            op_dict['stack_info'].append(tensor['full_info'])
            break
        op_dict["op_name"].append(tensor['full_op_name'])
        name_ele_list = tensor['full_op_name'].split(Const.SEP)
        name_to_struct_mapping = {
            Const.INPUT: CompareConst.INPUT_STRUCT,
            Const.KWARGS: CompareConst.KWARGS_STRUCT,
            Const.OUTPUT: CompareConst.OUTPUT_STRUCT
        }
        for name_key, struct_key in name_to_struct_mapping.items():
            if name_key in name_ele_list:
                if dump_mode == Const.MD5:
                    op_dict.get(struct_key).append((tensor[Const.DTYPE], tensor[Const.SHAPE], tensor[Const.MD5]))
                else:
                    op_dict.get(struct_key).append((tensor[Const.DTYPE], tensor[Const.SHAPE]))
                break
        op_dict[Const.SUMMARY].append([tensor[Const.MAX], tensor[Const.MIN], tensor[Const.MEAN], tensor[Const.NORM]])

        if dump_mode == Const.ALL:
            op_dict["data_name"].append(tensor['data_name'])
            data_name = op_dict["data_name"][-1].rsplit(Const.SEP, 1)[0]
            if data_name != "-1":
                op_dict["op_name"][-1] = data_name

    if not op_dict[CompareConst.KWARGS_STRUCT]:
        del op_dict[CompareConst.KWARGS_STRUCT]
    return op_dict if op_dict["op_name"] else {}


def _compare_parser(parser):
    parser.add_argument("-i", "--input_path", dest="input_path", type=str,
                        help="<Required> The compare input path, a dict json.", required=True)
    parser.add_argument("-o", "--output_path", dest="output_path", type=str,
                        help="<Required> The compare task result out path.", required=True)
    parser.add_argument("-s", "--stack_mode", dest="stack_mode", action="store_true",
                        help="<optional> Whether to save stack info.", required=False)
    parser.add_argument("-c", "--compare_only", dest="compare_only", action="store_true",
                        help="<optional> Whether to give advisor.", required=False)
    parser.add_argument("-f", "--fuzzy_match", dest="fuzzy_match", action="store_true",
                        help="<optional> Whether to perform a fuzzy match on the api name.", required=False)
    parser.add_argument("-cm", "--cell_mapping", dest="cell_mapping", type=str, nargs='?', const=True,
                        help="<optional> The cell mapping file path.", required=False)
    parser.add_argument("-am", "--api_mapping", dest="api_mapping", type=str, nargs='?', const=True,
                        help="<optional> The api mapping file path.", required=False)
    parser.add_argument("-dm", "--data_mapping", dest="data_mapping", type=str,
                        help="<optional> The data mapping file path.", required=False)
    parser.add_argument("-lm", "--layer_mapping", dest="layer_mapping", type=str, nargs='?', const=True,
                        help="<optional> The layer mapping file path.", required=False)
