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
import re
import math
import zlib
from dataclasses import dataclass

import numpy as np

from msprobe.core.common.const import Const, CompareConst, FileCheckConst
from msprobe.core.common.utils import CompareException, check_regex_prefix_format_valid, logger, safe_get_value
from msprobe.core.common.file_utils import check_file_or_directory_path


def extract_json(dirname, stack_json=False):
    json_path = ''
    for filename in os.listdir(dirname):
        target_file_name = 'stack.json' if stack_json else 'dump.json'
        if filename == target_file_name:
            json_path = os.path.join(dirname, filename)
            break

    # Provide robustness on invalid directory inputs
    if not json_path:
        if stack_json:
            logger.warning(f'stack.json is not found in dump dir {dirname}.')
        else:
            logger.error(f'dump.json is not found in dump dir {dirname}.')
            raise CompareException(CompareException.NO_DUMP_FILE_ERROR)
    return json_path


def set_stack_json_path(input_param):
    npu_data_dir = os.path.dirname(input_param.get("npu_json_path"))
    stack_path = extract_json(npu_data_dir, stack_json=True)
    input_param["stack_json_path"] = stack_path if stack_path else None
    return bool(stack_path)


def check_and_return_dir_contents(dump_dir, prefix):
    """
    check the given dump dir and validate files in dump dir by using the given prefix patterns to build a
    pattern: ^{prefix}(?:0|[1-9][0-9]*)?$

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
    pattern = re.compile(rf'^{prefix}(?:0|[1-9][0-9]*)?$')
    for name in contents:
        if not pattern.match(name):
            logger.error(
                f"dump_dir contains '{name}'. Expected '{prefix}'. This name is not in the format of dump "
                f"output. Please check and delete irrelevant files in {dump_dir} and try again."
            )
            raise CompareException(CompareException.INVALID_PATH_ERROR)
    return contents


def rename_api(npu_name, process):
    """
    原api： {api_type}.{api_name}.{API调用次数}.{前向反向}.{input/output}.{参数序号}
    rename后： {api_type}.{api_name}.{input/output}.{参数序号}
    """
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
    if Const.PARAMS_GRAD in op_name.split(Const.SEP):
        op_parsed_list = op_item_parse(op_data, op_name)
    else:
        op_parsed_list = []
        for name in CompareConst.IO_NAME_MAPPING:
            if name in op_data:
                op_parsed_list.extend(op_item_parse(op_data[name], op_name + CompareConst.IO_NAME_MAPPING[name]))
    return op_parsed_list


def op_item_parse(op_data, op_name: str, depth: int = 0) -> list:
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

    if depth > Const.MAX_DEPTH:
        logger.error(f'parse of api/module of {op_name} exceeds the recursion limit.')
        raise CompareException(CompareException.RECURSION_LIMIT_ERROR)

    if op_data is None:
        return [default_item]
    elif not op_data:
        return []

    item_list = []
    if isinstance(op_data, list):
        for i, data in enumerate(op_data):
            if Const.PARAMS_GRAD not in op_name.split(Const.SEP):
                item_list.extend(op_item_parse(data, op_name + Const.SEP + str(i), depth + 1))
            else:
                item_list.extend(op_item_parse(data, op_name, depth + 1))
    elif isinstance(op_data, dict):
        if is_leaf_data(op_data):
            return [gen_op_item(op_data, op_name)]
        for sub_name, sub_data in op_data.items():
            item_list.extend(op_item_parse(sub_data, op_name + Const.SEP + str(sub_name), depth + 1))
    return item_list


def is_leaf_data(op_data):
    return 'type' in op_data and isinstance(op_data['type'], str)


def gen_op_item(op_data, op_name):
    op_item = {}
    op_item.update(op_data)
    data_name = op_data.get('data_name') if op_data.get('data_name') else '-1'  # 如果是""也返回-1
    op_item['data_name'] = data_name
    op_item['full_op_name'] = data_name.rsplit(Const.SEP, 1)[0] if data_name != '-1' else op_name

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
        elif op_item.get('type') == 'ellipsis':
            op_item['dtype'] = op_data.get('type')
            op_item['shape'] = '[]'
            for i in params:
                op_item[i] = op_data.get('value')
        elif op_item.get('type') == 'torch.ProcessGroup':
            op_item['dtype'] = op_data.get('type')
            op_item['shape'] = '[]'
            for i in params:
                op_item[i] = str(op_data.get('group_ranks'))
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


@dataclass
class ApiItemInfo:
    name: str
    struct: tuple
    stack_info: list


def stack_column_process(result_item, has_stack, index, key, npu_stack_info):
    if has_stack and index == 0 and key == CompareConst.INPUT_STRUCT:
        result_item.extend(npu_stack_info)
    else:
        result_item.append(CompareConst.NONE)
    return result_item


def result_item_init(n_info, b_info, dump_mode):
    n_len = len(n_info.struct)
    b_len = len(b_info.struct)
    struct_long_enough = (n_len > 2 and b_len > 2) if dump_mode == Const.MD5 else (n_len > 1 and b_len > 1)
    if struct_long_enough:
        result_item = [
            n_info.name, b_info.name, n_info.struct[0], b_info.struct[0], n_info.struct[1], b_info.struct[1]
        ]
        if dump_mode == Const.MD5:
            md5_compare_result = CompareConst.PASS if n_info.struct[2] == b_info.struct[2] else CompareConst.DIFF
            result_item.extend([n_info.struct[2], b_info.struct[2], md5_compare_result])
        elif dump_mode == Const.SUMMARY:
            result_item.extend([" "] * 8)
        else:
            result_item.extend([" "] * 5)
    else:
        err_msg = "index out of bounds error will occur in result_item_init, please check!\n" \
                  f"npu_info_struct is {n_info.struct}\n" \
                  f"bench_info_struct is {b_info.struct}"
        logger.error(err_msg)
        raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR)
    return result_item


def count_struct(op_dict):
    parts = [
        CompareConst.OP_NAME,
        CompareConst.INPUT_STRUCT,
        CompareConst.OUTPUT_STRUCT,
        CompareConst.PARAMS_STRUCT,
        CompareConst.PARAMS_GRAD_STRUCT
    ]
    lengths = [len(op_dict.get(part, [])) for part in parts]
    num = lengths[0]
    if num != sum(lengths[1:]):
        logger.error(f"Length of names and structs of op_dict not match. Please check! op_dict: {op_dict}")
        raise CompareException(CompareException.NAMES_STRUCTS_MATCH_ERROR)
    return tuple(lengths)


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
            n_name = safe_get_value(n_dict, n_start + index, "n_dict", key="op_name")
            b_name = safe_get_value(b_dict, b_start + index, "b_dict", key="op_name")
            n_struct = safe_get_value(n_dict, index, "n_dict", key=key)
            b_struct = safe_get_value(b_dict, index, "b_dict", key=key)
            err_msg = ""

            npu_info = ApiItemInfo(n_name, n_struct, npu_stack_info)
            bench_info = ApiItemInfo(b_name, b_struct, bench_stack_info)
            result_item = result_item_init(npu_info, bench_info, dump_mode)

            if dump_mode == Const.MD5:
                result_item = stack_column_process(result_item, has_stack, index, key, npu_stack_info)
                result.append(result_item)
                continue

            npu_summary_data = safe_get_value(n_dict, n_start + index, "n_dict", key=CompareConst.SUMMARY)
            bench_summary_data = safe_get_value(b_dict, b_start + index, "b_dict", key=CompareConst.SUMMARY)
            result_item.extend(process_summary_data(npu_summary_data))
            result_item.extend(process_summary_data(bench_summary_data))

            if dump_mode == Const.SUMMARY:
                result_item, accuracy_check, err_msg = get_rela_diff_summary_mode(result_item, npu_summary_data,
                                                                                  bench_summary_data, err_msg)

            result_item.append(accuracy_check if dump_mode == Const.SUMMARY else CompareConst.ACCURACY_CHECK_YES)
            result_item.append(err_msg)
            result_item = stack_column_process(result_item, has_stack, index, key, npu_stack_info)
            if dump_mode == Const.ALL:
                result_item.append(safe_get_value(npu_data_name, n_start + index, "npu_data_name"))

            result.append(result_item)

        if n_len > b_len:
            for index in range(b_len, n_len):
                try:
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
                except IndexError as e:
                    err_msg = "index out of bounds error occurs, please check!\n" \
                              f"n_dict is {n_dict}"
                    logger.error(err_msg)
                    raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from e

                err_msg = ""
                result_item.append(CompareConst.ACCURACY_CHECK_YES)
                result_item.append(err_msg)
                result_item = stack_column_process(result_item, has_stack, index, key, npu_stack_info)
                if dump_mode == Const.ALL:
                    result_item.append(safe_get_value(npu_data_name, n_start + index, "npu_data_name"))

                result.append(result_item)

    n_num, n_num_input, n_num_output, n_num_params, n_num_params_grad = count_struct(n_dict)
    b_num, b_num_input, b_num_output, b_num_params, b_num_params_grad = count_struct(b_dict)

    get_accuracy_core(0, n_num_input, 0, b_num_input, CompareConst.INPUT_STRUCT)
    get_accuracy_core(n_num_input + n_num_output, n_num_params, b_num_input + b_num_output, b_num_params,
                      CompareConst.PARAMS_STRUCT)
    get_accuracy_core(n_num_input, n_num_output, b_num_input, b_num_output, CompareConst.OUTPUT_STRUCT)
    get_accuracy_core(n_num_input + n_num_output + n_num_params, n_num_params_grad,
                      b_num_input + b_num_output + b_num_params, b_num_params_grad,
                      CompareConst.PARAMS_GRAD_STRUCT)


def append_stack_info(result_item, npu_stack_info, index):
    """添加堆栈信息到 result_item"""
    if npu_stack_info and index == 0:
        result_item.extend(npu_stack_info)
    else:
        result_item.append(CompareConst.NONE)


def get_un_match_accuracy(result, n_dict, dump_mode):
    npu_stack_info = n_dict.get("stack_info", None)
    bench_name, bench_type, bench_shape = CompareConst.N_A, CompareConst.N_A, CompareConst.N_A

    struct_to_index_mapping = {
        CompareConst.INPUT_STRUCT: 0,
        CompareConst.OUTPUT_STRUCT: 0,
        CompareConst.PARAMS_STRUCT: 0,
        CompareConst.PARAMS_GRAD_STRUCT: 0
    }

    op_name_list = n_dict.get(CompareConst.OP_NAME)
    summary_list = n_dict.get(Const.SUMMARY)
    data_name_list = n_dict.get('data_name')
    op_name_reorder, summary_reorder, _ = reorder_op_x_list(op_name_list,
                                                            summary_list,
                                                            data_name_list)
    for index, n_name in enumerate(op_name_reorder):
        _, state = get_name_and_state(n_name)
        struct_key = CompareConst.STATE_TO_STRUCT_MAPPING.get(state)
        if not struct_key:
            continue
        n_struct = safe_get_value(n_dict, struct_to_index_mapping.get(struct_key), "n_dict", key=struct_key)
        struct_to_index_mapping[struct_key] += 1

        try:
            result_item = [n_name, bench_name, n_struct[0], bench_type, n_struct[1], bench_shape]
        except IndexError as e:
            err_msg = "index out of bounds error occurs, please check!\n" \
                      f"op_name of n_dict is {n_dict['op_name']}\n" \
                      f"input_struct of n_dict is {n_dict[CompareConst.INPUT_STRUCT]}\n" \
                      f"output_struct of n_dict is {n_dict[CompareConst.OUTPUT_STRUCT]}"
            logger.error(err_msg)
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from e

        if dump_mode == Const.MD5:
            result_item.extend([CompareConst.N_A] * 3)
            append_stack_info(result_item, npu_stack_info, index)
            result.append(result_item)
            continue
        if dump_mode == Const.SUMMARY:
            result_item.extend([CompareConst.N_A] * 8)
        if dump_mode == Const.ALL:
            result_item.extend([CompareConst.N_A] * 5)

        npu_summary_data = safe_get_value(summary_reorder, index, "summary_reorder")
        bench_summary_data = [CompareConst.N_A] * 4
        result_item.extend(npu_summary_data)
        result_item.extend(bench_summary_data)
        err_msg = CompareConst.NO_BENCH
        accuracy_check_res = CompareConst.N_A
        result_item.append(accuracy_check_res)
        result_item.append(err_msg)
        append_stack_info(result_item, npu_stack_info, index)
        if dump_mode == Const.ALL and result_item[1] == CompareConst.N_A:
            result_item.extend(["-1"])
        result.append(result_item)


def merge_tensor(tensor_list, dump_mode):
    op_dict = {}
    op_dict["op_name"] = []
    op_dict[CompareConst.INPUT_STRUCT] = []
    op_dict[CompareConst.KWARGS_STRUCT] = []
    op_dict[CompareConst.OUTPUT_STRUCT] = []
    op_dict[CompareConst.PARAMS_STRUCT] = []
    op_dict[CompareConst.PARAMS_GRAD_STRUCT] = []
    op_dict[Const.SUMMARY] = []
    op_dict["stack_info"] = []

    if dump_mode == Const.ALL:
        op_dict["data_name"] = []

    for tensor in tensor_list:
        # A dict(len=2) with 'full_op_name' and 'full_info' is added to the tensor only if self.stack_mode is True
        if len(tensor) == 2:
            op_dict['stack_info'].append(tensor['full_info'])
            break

        op_dict["op_name"].append(tensor['full_op_name'])

        _, state = get_name_and_state(tensor['full_op_name'])
        struct_key = CompareConst.STATE_TO_STRUCT_MAPPING.get(state)
        if not struct_key:
            continue
        if dump_mode == Const.MD5:
            op_dict.get(struct_key).append((tensor[Const.DTYPE], tensor[Const.SHAPE], tensor[Const.MD5]))
        else:
            op_dict.get(struct_key).append((tensor[Const.DTYPE], tensor[Const.SHAPE]))
        op_dict[Const.SUMMARY].append([tensor[Const.MAX], tensor[Const.MIN], tensor[Const.MEAN], tensor[Const.NORM]])

        if dump_mode == Const.ALL:
            op_dict["data_name"].append(tensor['data_name'])

    if not op_dict[CompareConst.KWARGS_STRUCT]:
        del op_dict[CompareConst.KWARGS_STRUCT]
    return op_dict if op_dict["op_name"] else {}


def print_compare_ends_info():
    total_len = len(CompareConst.COMPARE_ENDS_SUCCESSFULLY) + Const.FILL_CHAR_NUMS
    logger.info('*' * total_len)
    logger.info(f"*{CompareConst.COMPARE_ENDS_SUCCESSFULLY.center(total_len - 2)}*")
    logger.info('*' * total_len)


def table_value_is_valid(value: str) -> bool:
    if not isinstance(value, str):
        return True
    try:
        # -1.00 or +1.00 should be consdiered as digit numbers
        float(value)
    except ValueError:
        # otherwise, they will be considered as formular injections
        return not bool(re.compile(FileCheckConst.CSV_BLACK_LIST).search(value))
    return True


def get_name_and_state(name):
    """
    Get api/module name and state
    example:
    name = 'conv2d.forward.1.input.0'
    return: ('conv2d.forward.1.', 'input')

    name = 'Functional.pad.0.backward.output.0'
    return: ('Functional.pad.0.backward.', 'output')

    state type: input, output, kwargs, parameters, parameters_grad
    """
    if Const.PARAMS_GRAD in name.split(Const.SEP):
        return name.split(Const.PARAMS_GRAD)[0], Const.PARAMS_GRAD

    split = re.split(Const.REGEX_FORWARD_BACKWARD, name)
    api = f'{split[0]}.{split[1]}.'
    state_str = split[2]
    match = re.match(r'^(\d+\.)?(input|output|kwargs|parameters)\..+$', state_str)
    if not match:
        raise CompareException(f'Invalid name string: {name}')
    if match.group(1):
        api = f'{api}{match.group(1)}'
    state = match.group(2)
    return api, state


def reorder_op_name_list(op_name_list):
    if not op_name_list:
        return op_name_list

    parameters = []
    output = []
    parameters_grad = []
    others = []
    for x in op_name_list:
        state = get_name_and_state(x)[1]
        if state == Const.PARAMS:
            parameters.append(x)
        elif state == Const.OUTPUT:
            output.append(x)
        elif state == Const.PARAMS_GRAD:
            parameters_grad.append(x)
        else:
            others.append(x)
    # 合并others, parameters, 和output，确保parameters排在output前面
    op_name_reorder = others + parameters + output + parameters_grad
    return op_name_reorder


def reorder_op_x_list(op_name_list, summary_list, data_name_list):
    """对op_name, summary, data_name重新排序，把parameters放到input后output前，data_name由于统计量比对时，为None，单独处理"""
    if not op_name_list or not summary_list:
        return op_name_list, summary_list, data_name_list

    index_map = {name: index for index, name in enumerate(op_name_list)}

    op_name_reorder = reorder_op_name_list(op_name_list)
    summary_reorder = [summary_list[index_map.get(name)] for name in op_name_reorder]
    if data_name_list:
        data_name_reorder = [data_name_list[index_map.get(name)] for name in op_name_reorder]
    else:
        data_name_reorder = data_name_list

    return op_name_reorder, summary_reorder, data_name_reorder


def _compare_parser(parser):
    parser.add_argument("-i", "--input_path", dest="input_path", type=str,
                        help="<Required> The compare input path, a dict json.", required=True)
    parser.add_argument("-o", "--output_path", dest="output_path", type=str,
                        help="<Required> The compare task result out path. Default path: ./output",
                        required=False, default="./output", nargs="?", const="./output")
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
