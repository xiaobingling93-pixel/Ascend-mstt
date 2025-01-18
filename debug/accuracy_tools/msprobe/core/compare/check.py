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

from msprobe.core.common.log import logger
from msprobe.core.compare.utils import rename_api
from msprobe.core.common.utils import check_op_str_pattern_valid, CompareException
from msprobe.core.common.const import CompareConst, Const

dtype_mapping = {
    "Int8": "torch.int8",
    "UInt8": "torch.uint8",
    "Int16": "torch.int16",
    "UInt16": "torch.uint16",
    "Int32": "torch.int32",
    "UInt32": "torch.uint32",
    "Int64": "torch.int64",
    "UInt64": "torch.uint64",
    "Float16": "torch.float16",
    "Float32": "torch.float32",
    "Float64": "torch.float64",
    "Bool": "torch.bool",
    "BFloat16": "torch.bfloat16",
    "Complex64": "torch.complex64",
    "Complex128": "torch.complex128"
}


def compare_op_dict_struct(npu_dict, bench_dict):
    return all(npu_dict.get(key) == bench_dict.get(key) for key in CompareConst.STRUCT_COMPARE_KEY)


def check_struct_match(npu_dict, bench_dict):
    is_match = compare_op_dict_struct(npu_dict, bench_dict)
    if not is_match:
        struct_match_list = []
        try:
            for i, key in enumerate(CompareConst.STRUCT_COMPARE_KEY):
                # 首先额外检查input_struct是否空，input_struct不可能为空
                if i == 0 and (not npu_dict.get(key, []) or not bench_dict.get(key, [])):
                    return False
                struct_match_list.append(check_type_shape_match(npu_dict.get(key, []), bench_dict.get(key, [])))
        except CompareException as error:
            err_msg = f'index out of bounds error occurs in npu or bench api, please check!\n' \
                      f'npu_dict: {npu_dict}' \
                      f'bench_dict: {bench_dict}'
            logger.error(err_msg)
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from error
        is_match = all(struct_match_list)
    return is_match


def check_type_shape_match(npu_struct, bench_struct):
    """
    further check dtypes with a dtype mapping list when dtypes are not entirely consistent.
    """
    if len(npu_struct) != len(bench_struct):
        return False
    if not npu_struct and not bench_struct:
        return True

    struct_match = False
    for npu_type_shape, bench_type_shape in zip(npu_struct, bench_struct):
        try:
            npu_type = npu_type_shape[0]
            npu_shape = npu_type_shape[1]
            bench_type = bench_type_shape[0]
            bench_shape = bench_type_shape[1]
        except IndexError as error:
            logger.error(f'length of npu_type_shape: {npu_type_shape} and bench_type_shape: {bench_type_shape} '
                         f'should both be 2, please check!')
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from error
        shape_match = npu_shape == bench_shape
        type_match = npu_type == bench_type
        if not type_match:
            if ([npu_type, bench_type] in CompareConst.MS_TYPE) or ([npu_type, bench_type] in CompareConst.TORCH_TYPE):
                type_match = True
            else:
                type_match = False
        struct_match = shape_match and type_match
        if not struct_match:
            return False
    return struct_match


def check_graph_mode(a_op_name, b_op_name):
    if Const.ATEN in a_op_name and Const.ATEN not in b_op_name:
        return True
    if Const.ATEN not in a_op_name and Const.ATEN in b_op_name:
        return True
    return False


def fuzzy_check_op(npu_name_list, bench_name_list):
    # 先检查api里的item长度是否相等，如果不是parameters_grad, 必然有input或者output，长度不可能为0
    # 如果是parameters_grad, "parameters_grad"字段的字典不会是空字典，因此len>=1
    if len(npu_name_list) == 0 or len(bench_name_list) == 0 or len(npu_name_list) != len(bench_name_list):
        return False
    is_match = True
    for npu_name, bench_name in zip(npu_name_list, bench_name_list):
        is_match = fuzzy_check_name(npu_name, bench_name)
        if not is_match:
            break
    return is_match


def fuzzy_check_name(npu_name, bench_name):
    if Const.FORWARD in npu_name and Const.FORWARD in bench_name:
        is_match = rename_api(npu_name, Const.FORWARD) == rename_api(bench_name, Const.FORWARD)
    elif Const.BACKWARD in npu_name and Const.BACKWARD in bench_name:
        is_match = rename_api(npu_name, Const.BACKWARD) == rename_api(bench_name, Const.BACKWARD)
    else:
        is_match = npu_name == bench_name
    return is_match


def check_dump_json_str(op_data, op_name):
    input_list = op_data.get(Const.INPUT_ARGS, None) if op_data.get(Const.INPUT_ARGS, None) else op_data.get(
        Const.INPUT, None)
    input_kwargs = op_data.get(Const.INPUT_KWARGS, None)
    output_list = op_data.get(Const.OUTPUT, None)

    args = [input_list, input_kwargs, output_list]
    for arg in args:
        if not arg:
            continue
        if isinstance(arg, dict):
            check_json_key_value(arg, op_name)
        else:
            for ele in arg:
                if not ele:
                    continue
                check_json_key_value(ele, op_name)


def check_json_key_value(input_output, op_name, depth=0):
    if depth > Const.MAX_DEPTH:
        logger.error(f"string check of data info of {op_name} exceeds the recursion limit.")
        return
    if isinstance(input_output, list):
        for item in input_output:
            check_json_key_value(item, op_name, depth + 1)
    elif isinstance(input_output, dict):
        for key, value in input_output.items():
            if isinstance(value, dict):
                check_json_key_value(value, op_name, depth + 1)
            else:
                valid_key_value(key, value, op_name)


def valid_key_value(key, value, op_name):
    if key == "shape" and not isinstance(value, (list, tuple)):
        logger.error(f"shape of input or output of {op_name} is not list or tuple, please check!")
        raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)
    elif key == "requires_grad" and not isinstance(value, bool):
        logger.error(f"requires_grad of input or output of {op_name} is not bool, please check!")
        raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)
    else:
        check_op_str_pattern_valid(value, op_name)


def check_stack_json_str(stack_info, op_name):
    if isinstance(stack_info, list):
        for item in stack_info:
            check_op_str_pattern_valid(item, op_name, stack=True)
    else:
        logger.error(f"Expected stack_info to be a list, but got {type(stack_info).__name__} for '{op_name}'")
        raise CompareException(CompareException.INVALID_OBJECT_TYPE_ERROR)
