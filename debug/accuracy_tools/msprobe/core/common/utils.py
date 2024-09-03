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

from msprobe.core.common.file_utils import (FileOpen, check_file_or_directory_path)
from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.log import logger


device = collections.namedtuple('device', ['type', 'index'])
prefixes = ['api_stack', 'list', 'range', 'acl']


class CompareException(Exception):
    """
    Class for Accuracy Compare Exception
    """
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


    def __init__(self, code, error_info: str = ""):
        super(CompareException, self).__init__()
        self.code = code
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class DumpException(CompareException):
    pass


def check_mode_valid(mode, scope=None, api_list=None):
    if scope is None:
        scope = []
    if api_list is None:
        api_list = []
    if not isinstance(scope, list):
        raise ValueError("scope param set invalid, it's must be a list.")
    if not isinstance(api_list, list):
        raise ValueError("api_list param set invalid, it's must be a list.")
    mode_check = {
        Const.ALL: lambda: None,
        Const.RANGE: lambda:  ValueError("set_dump_switch, scope param set invalid, it's must be [start, end].") if len(scope) != 2 else None,
        Const.LIST: lambda:  ValueError("set_dump_switch, scope param set invalid, it's should not be an empty list.") if len(scope) == 0 else None,
        Const.STACK: lambda:  ValueError("set_dump_switch, scope param set invalid, it's must be [start, end] or [].") if len(scope) > 2 else None,
        Const.ACL: lambda:  ValueError("set_dump_switch, scope param set invalid, only one api name is supported in acl mode.") if len(scope) != 1 else None,
        Const.API_LIST: lambda:  ValueError("Current dump mode is 'api_list', but the content of api_list parameter is empty or valid.") if len(api_list) < 1 else None,
        Const.API_STACK: lambda: None,
    }
    if mode not in Const.DUMP_MODE:
        msg = "Current mode '%s' is not supported. Please use the field in %s" % \
              (mode, Const.DUMP_MODE)
        raise CompareException(CompareException.INVALID_DUMP_MODE, msg)

    if mode_check.get(mode)() is not None:
        raise mode_check.get(mode)()


def check_switch_valid(switch):
    if switch not in ["ON", "OFF"]:
        logger.error("Please set switch with 'ON' or 'OFF'.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)


def check_dump_mode_valid(dump_mode):
    if not isinstance(dump_mode, list):
        logger.warning("Please set dump_mode as a list.")
        dump_mode = [dump_mode]
    if not all(mode in ["all", "forward", "backward", "input", "output"] for mode in dump_mode):
        raise ValueError("Please set dump_mode as a list containing one or more of the following: 'all', 'forward', 'backward', 'input', 'output'.")
    if 'input' not in dump_mode and 'output' not in dump_mode:
        dump_mode.extend(['input', 'output'])
    if 'forward' not in dump_mode and 'backward' not in dump_mode:
        dump_mode.extend(['forward', 'backward'])
    if 'all' in dump_mode or set(["forward", "backward", "input", "output"]).issubset(set(dump_mode)):
        return ["forward", "backward", "input", "output"]
    return dump_mode


def check_summary_mode_valid(summary_mode):
    if summary_mode not in Const.SUMMARY_MODE:
        msg = "The summary_mode is not valid"
        raise CompareException(CompareException.INVALID_SUMMARY_MODE, msg)


def check_summary_only_valid(summary_only):
    if not isinstance(summary_only, bool):
        logger.error("Params summary_only only support True or False.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    return summary_only


def check_compare_param(input_param, output_path, summary_compare=False, md5_compare=False):
    if not (isinstance(input_param, dict) and isinstance(output_path, str)):
        logger.error("Invalid input parameters")
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



def check_configuration_param(stack_mode=False, auto_analyze=True, fuzzy_match=False):
    if not (isinstance(stack_mode, bool) and isinstance(auto_analyze, bool) and isinstance(fuzzy_match, bool)):
        logger.error("Invalid input parameters which should be only bool type.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)


def is_starts_with(string, prefix_list):
    return any(string.startswith(prefix) for prefix in prefix_list)


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


def get_dump_data_path(dump_dir):
    """
    Function Description:
        traverse directories and obtain the absolute path of dump data
    Parameter:
        dump_dir: dump data directory
    Return Value:
        dump data path,file is exist or file is not exist
    """
    dump_data_path = None
    file_is_exist = False

    check_file_or_directory_path(dump_dir, True)
    for dir_path, _, files in os.walk(dump_dir):
        if len(files) != 0:
            dump_data_path = dir_path
            file_is_exist = True
            break
        dump_data_path = dir_path
    return dump_data_path, file_is_exist


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


def parse_value_by_comma(value):
    """
    parse value by comma, like '1,2,4,8'
    """
    value_list = []
    value_str_list = value.split(Const.COMMA)
    for value_str in value_str_list:
        value_str = value_str.strip()
        if value_str.isdigit() or value_str == '-1':
            value_list.append(int(value_str))
        else:
            logger.error("please check your input shape.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
    return value_list


def add_time_as_suffix(name):
    return '{}_{}.csv'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))


def add_time_with_xlsx(name):
    return '{}_{}.xlsx'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))


def get_time():
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def format_value(value):
    return float('{:.12f}'.format(value))


def check_seed_all(seed, mode):
    if isinstance(seed, int):
        if seed < 0 or seed > Const.MAX_SEED_VALUE:
            logger.error(f"Seed must be between 0 and {Const.MAX_SEED_VALUE}.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
    else:
        logger.error(f"Seed must be integer.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    if not isinstance(mode, bool):
        logger.error(f"seed_all mode must be bool.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)


def get_process_rank(model):
    logger.info("Rank id is not provided. Trying to get the rank id of the model.")
    try:
        local_device = next(model.parameters()).device
    except StopIteration:
        logger.warning('There is no parameter in the model. Fail to get rank id.')
        return 0, False
    if local_device.type == 'cpu':
        logger.warning("Warning: the debugger is unable to get the rank id. "
            "This may cause the dumpped data to be corrupted in the "
            "case of distributed training. (You may ignore this if you are using only one card.) "
            "Transfer the model to npu or gpu before register_hook() to avoid this warning.")
        return 0, False
    else:
        return local_device.index, True


def generate_compare_script(dump_path, pkl_file_path, dump_switch_mode):
    template_path = os.path.join(os.path.dirname(__file__), "compare_script.template")
    pkl_dir = os.path.dirname(pkl_file_path)
    compare_script_path = os.path.join(pkl_dir, "compare_data.py")
    is_api_stack = "True" if dump_switch_mode == Const.API_STACK else "False"

    try:
        with FileOpen(template_path, 'r') as ftemp, \
           os.fdopen(os.open(compare_script_path, Const.WRITE_FLAGS, Const.WRITE_MODES), 'w+') as fout:
            code_temp = ftemp.read()
            fout.write(code_temp % (pkl_file_path, dump_path, is_api_stack))
    except OSError:
        logger.error(f"Failed to open file. Please check file {template_path} or path {pkl_dir}.")

    logger.info(f"Generate compare script successfully which is {compare_script_path}.")


def check_inplace_op(prefix):
    if len(prefix) > Const.DISTRIBUTED_PREFIX_LENGTH:
        return False
    match_op = re.findall(r"Distributed\.(.+?)\.\d", prefix)
    op_name = match_op[0] if match_op else None
    return op_name in Const.INPLACE_LIST


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
