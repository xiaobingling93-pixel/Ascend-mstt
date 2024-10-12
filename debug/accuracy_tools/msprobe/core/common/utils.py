#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""
import collections
import os
import re
import subprocess
import time
import json
from datetime import datetime, timezone

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


def check_compare_param(input_param, output_path, summary_compare=False, md5_compare=False):
    if not isinstance(input_param, dict):
        logger.error(f"Invalid input parameter 'input_param', the expected type dict but got {type(input_param)}.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    if not isinstance(output_path, str):
        logger.error(f"Invalid input parameter 'output_path', the expected type str but got {type(output_path)}.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)

    check_file_or_directory_path(input_param.get("npu_json_path"), False)
    check_file_or_directory_path(input_param.get("bench_json_path"), False)
    check_file_or_directory_path(input_param.get("stack_json_path"), False)
    if not summary_compare and not md5_compare:
        check_file_or_directory_path(input_param.get("npu_dump_data_dir"), True)
        check_file_or_directory_path(input_param.get("bench_dump_data_dir"), True)
    check_file_or_directory_path(output_path, True)

    with FileOpen(input_param.get("npu_json_path"), "r") as npu_json, \
            FileOpen(input_param.get("bench_json_path"), "r") as bench_json, \
            FileOpen(input_param.get("stack_json_path"), "r") as stack_json:
        check_json_file(input_param, npu_json, bench_json, stack_json)


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
            print(line)
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
            elif 'md5' in data[key_op][api_info]:
                return True
    return False


def struct_json_get(input_param, framework):
    if framework == Const.PT_FRAMEWORK:
        prefix = "bench"
    elif framework == Const.MS_FRAMEWORK:
        prefix = "npu"
    else:
        logger.error("Error framework found.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)

    frame_json_path = input_param.get(f"{prefix}_json_path", None)
    if not frame_json_path:
        logger.error(f"Please check the json path is valid.")
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    directory = os.path.dirname(frame_json_path)
    check_file_or_directory_path(directory, True)
    stack_json = os.path.join(directory, "stack.json")
    construct_json = os.path.join(directory, "construct.json")

    stack = load_json(stack_json)
    construct = load_json(construct_json)
    return stack, construct


def task_dumppath_get(input_param):
    npu_path = input_param.get("npu_json_path", None)
    bench_path = input_param.get("bench_json_path", None)
    if not npu_path or not bench_path:
        logger.error(f"Please check the json path is valid.")
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    with FileOpen(npu_path, 'r') as npu_f:
        npu_json_data = json.load(npu_f)
    with FileOpen(bench_path, 'r') as bench_f:
        bench_json_data = json.load(bench_f)
    if npu_json_data['task'] != bench_json_data['task']:
        logger.error(f"Please check the dump task is consistent.")
        raise CompareException(CompareException.INVALID_TASK_ERROR)
    if npu_json_data['task'] == Const.TENSOR:
        summary_compare = False
        md5_compare = False
    elif npu_json_data['task'] == Const.STATISTICS:
        md5_compare = md5_find(npu_json_data['data'])
        if md5_compare:
            summary_compare = False
        else:
            summary_compare = True
    else:
        logger.error(f"Compare is not required for overflow_check or free_benchmark.")
        raise CompareException(CompareException.INVALID_TASK_ERROR)
    input_param['npu_dump_data_dir'] = os.path.join(os.path.dirname(npu_path), Const.DUMP_TENSOR_DATA)
    input_param['bench_dump_data_dir'] = os.path.join(os.path.dirname(bench_path), Const.DUMP_TENSOR_DATA)
    return summary_compare, md5_compare


def get_header_index(header_name, summary_compare=False):
    if summary_compare:
        header = CompareConst.SUMMARY_COMPARE_RESULT_HEADER[:]
    else:
        header = CompareConst.COMPARE_RESULT_HEADER[:]
    if header_name not in header:
        logger.error(f"{header_name} not in data name")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    return header.index(header_name)


def convert_tuple(data):
    return data if isinstance(data, tuple) else (data, )


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
                               f'The string parameter for {obj} only supports formats like "3-5". Now string parameter for {obj} is "{step_or_rank}".')
    if all(Const.STEP_RANK_MAXIMUM_RANGE[0] <= b <= Const.STEP_RANK_MAXIMUM_RANGE[1] for b in borderlines):
        if borderlines[0] <= borderlines[1]:
            continual_step_or_rank = list(range(borderlines[0], borderlines[1] + 1))
        else:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, 
                               f'For the hyphen(-) in {obj}, the left boundary ({borderlines[0]}) cannot be greater than the right boundary ({borderlines[1]}).')
    else:
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, 
                               f"The boundaries must fall within the range of [{Const.STEP_RANK_MAXIMUM_RANGE[0]}, {Const.STEP_RANK_MAXIMUM_RANGE[1]}].")
    return continual_step_or_rank


def get_real_step_or_rank(step_or_rank_input, obj):
    if obj not in [Const.STEP, Const.RANK]:
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, 
                               f"Only support parsing {[Const.STEP, Const.RANK]}, the current parsing object is {obj}.")
    if step_or_rank_input is None:
        return []
    if not isinstance(step_or_rank_input, list):
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, f"{obj} is invalid, it should be a list")
    real_step_or_rank = []
    for element in step_or_rank_input:
        if not isinstance(element, (int, str)):
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, 
                                   f"{obj} element {element} must be an integer or string.")
        if isinstance(element, int) and element < 0:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, 
                                   f"Each element of {obj} must be non-negative, currently it is {element}.")
        if isinstance(element, int) and Const.STEP_RANK_MAXIMUM_RANGE[0] <= element <= Const.STEP_RANK_MAXIMUM_RANGE[1]:
            real_step_or_rank.append(element)
        elif isinstance(element, str) and Const.HYPHEN in element:
            continual_step_or_rank = get_step_or_rank_from_string(element, obj)
            real_step_or_rank.extend(continual_step_or_rank)
    real_step_or_rank = list(set(real_step_or_rank))
    real_step_or_rank.sort()
    return real_step_or_rank


def check_seed_all(seed, mode):
    if isinstance(seed, int):
        if seed < 0 or seed > Const.MAX_SEED_VALUE:
            logger.error(f"Seed must be between 0 and {Const.MAX_SEED_VALUE}.")
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    else:
        logger.error("Seed must be integer.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if not isinstance(mode, bool):
        logger.error("seed_all mode must be bool.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
