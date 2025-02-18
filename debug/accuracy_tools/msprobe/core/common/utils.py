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

import collections
import os
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import wraps

import numpy as np

from msprobe.core.common.file_utils import (FileOpen, check_file_or_directory_path, load_json)
from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException


device = collections.namedtuple('device', ['type', 'index'])
prefixes = ['api_stack', 'list', 'range', 'acl']


class MsprobeBaseException(Exception):
    """
    Base class for all custom exceptions.
    """
    # 所有的错误代码
    NONE_ERROR = 0
    INVALID_PATH_ERROR = 1
    OPEN_FILE_ERROR = 2
    CLOSE_FILE_ERROR = 3
    READ_FILE_ERROR = 4
    WRITE_FILE_ERROR = 5
    INVALID_FILE_ERROR = 6
    PERMISSION_ERROR = 7
    INDEX_OUT_OF_BOUNDS_ERROR = 8
    NO_DUMP_FILE_ERROR = 9
    INVALID_DATA_ERROR = 10
    INVALID_PARAM_ERROR = 11
    INVALID_DUMP_RATIO = 12
    INVALID_DUMP_FILE = 13
    UNKNOWN_ERROR = 14
    INVALID_DUMP_MODE = 15
    PARSE_FILE_ERROR = 16
    INVALID_COMPARE_MODE = 17
    OVER_SIZE_FILE_ERROR = 18
    INVALID_SUMMARY_MODE = 19
    INVALID_TASK_ERROR = 20
    DETACH_ERROR = 21
    INVALID_OBJECT_TYPE_ERROR = 22
    INVALID_CHAR_ERROR = 23
    RECURSION_LIMIT_ERROR = 24
    INVALID_ATTRIBUTE_ERROR = 25
    OUTPUT_HOOK_ERROR = 26
    INPUT_HOOK_ERROR = 27
    FUNCTION_CALL_ERROR = 28
    FORWARD_DATA_COLLECTION_ERROR = 29
    BACKWARD_DATA_COLLECTION_ERROR = 30
    INVALID_KEY_ERROR = 31
    MISSING_HEADER_ERROR = 32
    MERGE_COMPARE_RESULT_ERROR = 33
    NAMES_STRUCTS_MATCH_ERROR = 34
    INVALID_STATE_ERROR = 35

    def __init__(self, code, error_info: str = ""):
        super(MsprobeBaseException, self).__init__()
        self.code = code
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class CompareException(MsprobeBaseException):
    """
    Class for Accuracy Compare Exception
    """

    def __init__(self, code, error_info: str = ""):
        super(CompareException, self).__init__(code, error_info)


class DumpException(MsprobeBaseException):
    """
    Class for Dump Exception
    """

    def __init__(self, code, error_info: str = ""):
        super(DumpException, self).__init__(code, error_info)

    def __str__(self):
        return f"Dump Error Code {self.code}: {self.error_info}"


def is_json_file(file_path):
    if isinstance(file_path, str) and file_path.lower().endswith('.json'):
        return True
    else:
        return False


def check_compare_param(input_param, output_path, dump_mode, stack_mode):
    if not isinstance(input_param, dict):
        logger.error(f"Invalid input parameter 'input_param', the expected type dict but got {type(input_param)}.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    if not isinstance(output_path, str):
        logger.error(f"Invalid input parameter 'output_path', the expected type str but got {type(output_path)}.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)

    def check_json_path(json_path_str):
        json_path = input_param.get(json_path_str)
        check_file_or_directory_path(json_path, False)
        json_type_check = is_json_file(json_path)
        if not json_type_check:
            logger.error(f"Invalid {json_path_str}: {json_path}, please check!")
            raise CompareException(CompareException.INVALID_PATH_ERROR)

    check_json_path("npu_json_path")
    check_json_path("bench_json_path")
    if stack_mode:
        check_json_path("stack_json_path")

    if dump_mode == Const.ALL:
        check_file_or_directory_path(input_param.get("npu_dump_data_dir"), True)
        check_file_or_directory_path(input_param.get("bench_dump_data_dir"), True)
    check_file_or_directory_path(output_path, True)

    with FileOpen(input_param.get("npu_json_path"), "r") as npu_json, \
            FileOpen(input_param.get("bench_json_path"), "r") as bench_json:
        _check_json(npu_json, input_param.get("npu_json_path"))
        _check_json(bench_json, input_param.get("bench_json_path"))
    if stack_mode:
        with FileOpen(input_param.get("stack_json_path"), "r") as stack_json:
            _check_json(stack_json, input_param.get("stack_json_path"))


def check_configuration_param(stack_mode=False, auto_analyze=True, fuzzy_match=False, is_print_compare_log=True):
    arg_list = [stack_mode, auto_analyze, fuzzy_match, is_print_compare_log]
    for arg in arg_list:
        if not isinstance(arg, bool):
            logger.error(f"Invalid input parameter, {arg} which should be only bool type.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)


def _check_json(json_file_handle, file_name):
    tensor_line = json_file_handle.readline()
    if not tensor_line:
        logger.error("dump file {} have empty line!".format(file_name))
        raise CompareException(CompareException.INVALID_DUMP_FILE)
    json_file_handle.seek(0, 0)


def check_json_file(input_param, npu_json, bench_json, stack_json):
    _check_json(npu_json, input_param.get("npu_json_path"))
    _check_json(bench_json, input_param.get("bench_json_path"))
    _check_json(stack_json, input_param.get("stack_json_path"))


def check_regex_prefix_format_valid(prefix):
    """
        validate the format of the regex prefix

    Args:
        prefix (str): The prefix string to validate.

    Returns:
        no returns

    Raises:
        ValueError: if the prefix length exceeds Const.REGEX_PREFIX_MAX_LENGTH characters or the prefix do not match
        the given pattern Const.REGEX_PREFIX_PATTERN
    """
    if len(prefix) > Const.REGEX_PREFIX_MAX_LENGTH:
        raise ValueError(f"Maximum length of prefix is {Const.REGEX_PREFIX_MAX_LENGTH}, while current length "
                         f"is {len(prefix)}")
    if not re.match(Const.REGEX_PREFIX_PATTERN, prefix):
        raise ValueError(f"prefix contains invalid characters, prefix pattern {Const.REGEX_PREFIX_PATTERN}")


def execute_command(cmd):
    """
    Function Description:
        run the following command
    Parameter:
        cmd: command
    Exception Description:
        when invalid command throw exception
    """
    logger.info('Execute command:%s' % cmd)
    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while process.poll() is None:
        line = process.stdout.readline()
        line = line.strip()
        if line:
            logger.info(line)
    if process.returncode != 0:
        logger.error('Failed to execute command:%s' % " ".join(cmd))
        raise CompareException(CompareException.INVALID_DATA_ERROR)


def add_time_as_suffix(name):
    return '{}_{}.csv'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))


def add_time_with_xlsx(name):
    return '{}_{}.xlsx'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))


def add_time_with_yaml(name):
    return '{}_{}.yaml'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))


def get_time():
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def format_value(value):
    return float('{:.12f}'.format(value))


def md5_find(data):
    for key_op in data:
        for api_info in data[key_op]:
            if isinstance(data[key_op][api_info], list):
                for data_detail in data[key_op][api_info]:
                    if data_detail and 'md5' in data_detail:
                        return True
            if isinstance(data[key_op][api_info], bool):
                continue
            elif data[key_op][api_info] and 'md5' in data[key_op][api_info]:
                return True
    return False


def detect_framework_by_dump_json(file_path):
    pattern_ms = r'"type":\s*"mindspore'
    pattern_pt = r'"type":\s*"torch'
    with FileOpen(file_path, 'r') as file:
        for line in file:
            if re.search(pattern_ms, line):
                return Const.MS_FRAMEWORK
            if re.search(pattern_pt, line):
                return Const.PT_FRAMEWORK
    logger.error(f"{file_path} must be based on the MindSpore or PyTorch framework.")
    raise CompareException(CompareException.INVALID_PARAM_ERROR)


def get_stack_construct_by_dump_json_path(dump_json_path):
    if not dump_json_path:
        logger.error("The path is empty. Please enter a valid path.")
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    directory = os.path.dirname(dump_json_path)
    check_file_or_directory_path(directory, True)
    stack_json = os.path.join(directory, "stack.json")
    construct_json = os.path.join(directory, "construct.json")

    stack = load_json(stack_json)
    construct = load_json(construct_json)
    return stack, construct


def set_dump_path(input_param):
    npu_path = input_param.get("npu_json_path", None)
    bench_path = input_param.get("bench_json_path", None)
    npu_path_valid = npu_path is not None and npu_path.endswith("dump.json")
    bench_path_valid = bench_path is not None and bench_path.endswith("dump.json")
    if not npu_path_valid or not bench_path_valid:
        logger.error(f"Please check the json path is valid. npu_path: {npu_path}, bench_path: {bench_path}")
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    input_param['npu_dump_data_dir'] = os.path.join(os.path.dirname(npu_path), Const.DUMP_TENSOR_DATA)
    input_param['bench_dump_data_dir'] = os.path.join(os.path.dirname(bench_path), Const.DUMP_TENSOR_DATA)


def get_dump_mode(input_param):
    npu_path = input_param.get("npu_json_path", None)
    bench_path = input_param.get("bench_json_path", None)
    npu_json_data = load_json(npu_path)
    bench_json_data = load_json(bench_path)

    npu_task = npu_json_data.get('task', None)
    bench_task = bench_json_data.get('task', None)

    if not npu_task or not bench_task:
        logger.error(f"Please check the dump task is correct, npu's task is {npu_task}, bench's task is {bench_task}.")
        raise CompareException(CompareException.INVALID_TASK_ERROR)

    if npu_task != bench_task:
        logger.error(f"Please check the dump task is consistent.")
        raise CompareException(CompareException.INVALID_TASK_ERROR)

    if npu_task == Const.TENSOR:
        return Const.ALL

    if npu_task == Const.STRUCTURE:
        return Const.STRUCTURE

    if npu_task == Const.STATISTICS:
        npu_md5_compare = md5_find(npu_json_data['data'])
        bench_md5_compare = md5_find(bench_json_data['data'])
        if npu_md5_compare == bench_md5_compare:
            return Const.MD5 if npu_md5_compare else Const.SUMMARY
        else:
            logger.error(f"Please check the dump task is consistent, "
                         f"dump mode of npu and bench should both be statistics or md5.")
            raise CompareException(CompareException.INVALID_TASK_ERROR)

    logger.error(f"Compare applies only to task is tensor or statistics")
    raise CompareException(CompareException.INVALID_TASK_ERROR)


def get_header_index(header_name, dump_mode):
    header = CompareConst.HEAD_OF_COMPARE_MODE.get(dump_mode)
    if not header:
        logger.error(f"{dump_mode} not in {CompareConst.HEAD_OF_COMPARE_MODE}")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    if header_name not in header:
        logger.error(f"{header_name} not in data name")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    return header.index(header_name)


def convert_tuple(data):
    return data if isinstance(data, tuple) else (data,)


def check_op_str_pattern_valid(string, op_name=None, stack=False):
    if isinstance(string, str) and is_invalid_pattern(string):
        if stack:
            message = f"stack info of {op_name} contains special characters, please check!"
        elif not op_name:
            message = f"api name contains special characters, please check!"
        else:
            message = f"data info of {op_name} contains special characters, please check!"
        logger.error(message)
        raise CompareException(CompareException.INVALID_CHAR_ERROR)


def is_invalid_pattern(string):
    pattern = Const.STRING_BLACKLIST
    return re.search(pattern, string)


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def print_tools_ends_info():
    total_len = len(Const.TOOL_ENDS_SUCCESSFULLY) + Const.FILL_CHAR_NUMS
    logger.info('*' * total_len)
    logger.info(f"*{Const.TOOL_ENDS_SUCCESSFULLY.center(total_len - 2)}*")
    logger.info('*' * total_len)


def get_step_or_rank_from_string(step_or_rank, obj):
    splited = step_or_rank.split(Const.HYPHEN)
    if len(splited) == 2:
        try:
            borderlines = int(splited[0]), int(splited[1])
        except (ValueError, IndexError) as e:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   "The hyphen(-) must start and end with decimal numbers.") from e
    else:
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                               f'The string parameter for {obj} only supports formats like "3-5". '
                               f'Now string parameter for {obj} is "{step_or_rank}".')
    if all(Const.STEP_RANK_MINIMUM_VALUE <= b <= Const.STEP_RANK_MAXIMUM_VALUE for b in borderlines):
        if borderlines[0] <= borderlines[1]:
            continual_step_or_rank = list(range(borderlines[0], borderlines[1] + 1))
        else:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f'For the hyphen(-) in {obj}, the left boundary ({borderlines[0]}) cannot be '
                                   f'greater than the right boundary ({borderlines[1]}).')
    else:
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                               f"The boundaries must fall within the range of "
                               f"[{Const.STEP_RANK_MINIMUM_VALUE}, {Const.STEP_RANK_MAXIMUM_VALUE}].")
    return continual_step_or_rank


def get_real_step_or_rank(step_or_rank_input, obj):
    if obj not in [Const.STEP, Const.RANK]:
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                               f"Only support parsing {[Const.STEP, Const.RANK]}, the current parsing object is {obj}.")
    if step_or_rank_input is None:
        return []
    if not isinstance(step_or_rank_input, list):
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, f"{obj} is invalid, it should be a list")
    if len(step_or_rank_input) > Const.STEP_RANK_MAXIMUM_VALUE:
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                               f"{obj} is invalid, its length cannot exceed {Const.STEP_RANK_MAXIMUM_VALUE}")

    real_step_or_rank = []
    for element in step_or_rank_input:
        if not is_int(element) and not isinstance(element, str):
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   f"{obj} element {element} must be an integer or string.")
        if is_int(element):
            if not Const.STEP_RANK_MINIMUM_VALUE <= element <= Const.STEP_RANK_MAXIMUM_VALUE:
                raise MsprobeException(
                    MsprobeException.INVALID_PARAM_ERROR,
                    f"Each element of {obj} must be between {Const.STEP_RANK_MINIMUM_VALUE} and "
                    f"{Const.STEP_RANK_MAXIMUM_VALUE}, currently it is {element}."
                )
            real_step_or_rank.append(element)
            continue
        continual_step_or_rank = get_step_or_rank_from_string(element, obj)
        real_step_or_rank.extend(continual_step_or_rank)
    real_step_or_rank = list(set(real_step_or_rank))
    real_step_or_rank.sort()
    return real_step_or_rank


def check_seed_all(seed, mode, rm_dropout):
    if is_int(seed):
        if seed < 0 or seed > Const.MAX_SEED_VALUE:
            logger.error(f"Seed must be between 0 and {Const.MAX_SEED_VALUE}.")
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    else:
        logger.error("Seed must be integer.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if not isinstance(mode, bool):
        logger.error("seed_all mode must be bool.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if not isinstance(rm_dropout, bool):
        logger.error("The rm_dropout parameter must be bool.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)


def safe_get_value(container, index, container_name, key=None):
    try:
        # 处理字典情况
        if isinstance(container, dict):
            return container.get(key)[index]
        # 处理列表、元组、numpy情况
        elif isinstance(container, (list, tuple, np.ndarray)):
            return container[index]
        else:
            err_msg = f"Unsupported container type for '{container_name}': {type(container)}"
            logger.error(err_msg)
            raise MsprobeBaseException(MsprobeBaseException.INVALID_OBJECT_TYPE_ERROR)
    except IndexError as e:
        err_msg = "index out of bounds error occurs, please check!\n" \
                  f"{container_name} is {container}\n" \
                  f"index is {index}"
        logger.error(err_msg)
        raise MsprobeBaseException(MsprobeBaseException.INDEX_OUT_OF_BOUNDS_ERROR) from e
    except TypeError as e:
        err_msg = "wrong type, please check!\n" \
                  f"{container_name} is {container}\n" \
                  f"index is {index}\n" \
                  f"key is {key}"
        logger.error(err_msg)
        raise MsprobeBaseException(MsprobeBaseException.INVALID_OBJECT_TYPE_ERROR) from e


# 记录工具函数递归的深度
recursion_depth = defaultdict(int)


# 装饰一个函数，当函数递归调用超过限制时，抛出异常并打印函数信息。
def recursion_depth_decorator(func_info):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_id = id(func)
            recursion_depth[func_id] += 1
            if recursion_depth[func_id] > Const.MAX_DEPTH:
                msg = f"call {func_info} exceeds the recursion limit."
                logger.error_log_with_exp(
                    msg,
                    MsprobeException(
                        MsprobeException.RECURSION_LIMIT_ERROR, msg
                    ),
                )
            try:
                result = func(*args, **kwargs)
            finally:
                recursion_depth[func_id] -= 1
            return result

        return wrapper

    return decorator


def check_str_param(param):
    if not re.match(Const.REGEX_PREFIX_PATTERN, param):
        logger.error('The parameter {} contains special characters.'.format(param))
        raise MsprobeBaseException(MsprobeBaseException.INVALID_CHAR_ERROR)


class DumpPathAggregation:
    dump_file_path = None
    stack_file_path = None
    construct_file_path = None
    dump_tensor_data_dir = None
    free_benchmark_file_path = None
    debug_file_path = None