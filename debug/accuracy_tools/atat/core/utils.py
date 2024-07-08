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
import shutil
import stat
import subprocess
import sys
import time
import json
from json.decoder import JSONDecodeError
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from .file_check_util import FileOpen, FileChecker, FileCheckConst


device = collections.namedtuple('device', ['type', 'index'])
prefixes = ['api_stack', 'list', 'range', 'acl']


class Const:
    """
    Class for const
    """
    MODEL_TYPE = ['.onnx', '.pb', '.om']
    DIM_PATTERN = r"^(-?[0-9]+)(,-?[0-9]+)*"
    REGEX_PREFIX_MAX_LENGTH = 20
    REGEX_PREFIX_PATTERN = r"^[a-zA-Z0-9_-]+$"
    SEMICOLON = ";"
    COLON = ":"
    EQUAL = "="
    COMMA = ","
    DOT = "."
    DUMP_RATIO_MAX = 100
    SUMMERY_DATA_NUMS = 256
    FLOAT_EPSILON = np.finfo(float).eps
    SUPPORT_DUMP_MODE = ['api', 'acl']
    ON = 'ON'
    OFF = 'OFF'
    BACKWARD = 'backward'
    FORWARD = 'forward'
    PRE_FORWARD = "pre_forward"

    # dump mode
    ALL = "all"
    LIST = "list"
    RANGE = "range"
    STACK = "stack"
    ACL = "acl"
    API_LIST = "api_list"
    API_STACK = "api_stack"
    DUMP_MODE = [ALL, LIST, RANGE, STACK, ACL, API_LIST, API_STACK]
    AUTO = "auto"
    ONLINE_DUMP_MODE = [ALL, LIST, AUTO, OFF]
    SUMMARY = "summary"
    MD5 = "md5"
    SUMMARY_MODE = [ALL, SUMMARY, MD5]

    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR

    PKL_SUFFIX = ".pkl"
    NUMPY_SUFFIX = ".npy"
    ONE_GB = 1 * 1024 * 1024 * 1024
    TEN_GB = 10 * 1024 * 1024 * 1024
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    FILE_NAME_LENGTH = 255
    DIRECTORY_LENGTH = 4096
    DISTRIBUTED_PREFIX_LENGTH = 60
    SUMMARY_COLUMN_NUM = 6
    STACK_COLUMN_NUM = 2
    # env dump path
    ASCEND_WORK_PATH = "ASCEND_WORK_PATH"
    DUMP_DIR = "dump_data"

    ENV_ENABLE = "1"
    ENV_DISABLE = "0"

    MAX_SEED_VALUE = 2**32 - 1

    INPLACE_LIST = ["broadcast", "all_reduce", "reduce", "all_gather", "gather", "scatter", "reduce_scatter",
                    "_reduce_scatter_base", "_all_gather_base"]

    TASK_LIST = ["tensor", "statistics", "overflow_check", "free_benchmark"]
    LEVEL_LIST = ["L0", "L1", "L2", "mix"]
    STATISTICS = "statistics"
    TENSOR = "tensor"
    OVERFLOW_CHECK = "overflow_check"
    FREE_BENCHMARK = "free_benchmark"

class CompareConst:
    """
    Class for compare module const
    """
    # compare result column name
    NPU_NAME = "NPU Name"
    BENCH_NAME = "Bench Name"
    NPU_DTYPE = "NPU Dtype"
    BENCH_DTYPE = "Bench Dtype"
    NPU_SHAPE = "NPU Tensor Shape"
    BENCH_SHAPE = "Bench Tensor Shape"
    NPU_MAX = "NPU max"
    NPU_MIN = "NPU min"
    NPU_MEAN = "NPU mean"
    NPU_NORM = "NPU l2norm"
    BENCH_MAX = "Bench max"
    BENCH_MIN = "Bench min"
    BENCH_MEAN = "Bench mean"
    BENCH_NORM = "Bench l2norm"
    MAX_DIFF = "Max diff"
    MIN_DIFF = "Min diff"
    MEAN_DIFF = "Mean diff"
    NORM_DIFF = "L2norm diff"
    COSINE = "Cosine"
    MAX_ABS_ERR = "MaxAbsErr"
    MAX_RELATIVE_ERR = "MaxRelativeErr"
    MIN_RELATIVE_ERR = "MinRelativeErr"
    MEAN_RELATIVE_ERR = "MeanRelativeErr"
    NORM_RELATIVE_ERR = "NormRelativeErr"
    ACCURACY = "Accuracy Reached or Not"
    STACK = "NPU_Stack_Info"
    DATA_NAME = "Data_name"
    ERROR_MESSAGE = "Err_message"
    ONE_THOUSANDTH_ERR_RATIO = "One Thousandth Err Ratio"
    FIVE_THOUSANDTHS_ERR_RATIO = "Five Thousandths Err Ratio"
    NPU_MD5 = "NPU MD5"
    BENCH_MD5 = "BENCH MD5"
    RESULT = "Result"

    COMPARE_RESULT_HEADER = [
        NPU_NAME, BENCH_NAME, NPU_DTYPE, BENCH_DTYPE, NPU_SHAPE, BENCH_SHAPE, COSINE, MAX_ABS_ERR, MAX_RELATIVE_ERR,
        ONE_THOUSANDTH_ERR_RATIO, FIVE_THOUSANDTHS_ERR_RATIO,
        NPU_MAX, NPU_MIN, NPU_MEAN, NPU_NORM, BENCH_MAX, BENCH_MIN, BENCH_MEAN, BENCH_NORM, ACCURACY, ERROR_MESSAGE
    ]

    SUMMARY_COMPARE_RESULT_HEADER = [
        NPU_NAME, BENCH_NAME, NPU_DTYPE, BENCH_DTYPE, NPU_SHAPE, BENCH_SHAPE, MAX_DIFF, MIN_DIFF, MEAN_DIFF, NORM_DIFF,
        MAX_RELATIVE_ERR, MIN_RELATIVE_ERR, MEAN_RELATIVE_ERR, NORM_RELATIVE_ERR,
        NPU_MAX, NPU_MIN, NPU_MEAN, NPU_NORM, BENCH_MAX, BENCH_MIN, BENCH_MEAN, BENCH_NORM, RESULT, ERROR_MESSAGE
    ]

    MD5_COMPARE_RESULT_HEADER = [
        NPU_NAME, BENCH_NAME, NPU_DTYPE, BENCH_DTYPE, NPU_SHAPE, BENCH_SHAPE, NPU_MD5, BENCH_MD5, RESULT
    ]

    # compare standard
    THOUSAND_RATIO_THRESHOLD = 0.001
    FIVE_THOUSAND_RATIO_THRESHOLD = 0.005
    COSINE_THRESHOLD = 0.9999

    # compare result data
    READ_NONE = 'No data'
    NAN = 'Nan'
    NONE = 'None'
    SHAPE_UNMATCH = 'shape unmatched'
    DTYPE_UNMATCH = 'dtype unmatched'
    PASS = 'Pass'
    WARNING = 'Warning'
    DIFF = 'Different'
    UNSUPPORTED = 'unsupported'

    # accuracy standards
    COS_THRESHOLD = 0.99
    MAX_ABS_ERR_THRESHOLD = 0.001
    COS_MAX_THRESHOLD = 0.9
    MAX_ABS_ERR_MAX_THRESHOLD = 1
    ACCURACY_CHECK_YES = "Yes"
    ACCURACY_CHECK_NO = "No"
    ACCURACY_CHECK_UNMATCH = "Unmatched"

    # error message
    NO_BENCH = "No bench data matched."

    # compare const
    FLOAT_TYPE = [np.half, np.single, float, np.double, np.float64, np.longdouble]

    # highlight xlsx color const
    RED = "FFFF0000"
    YELLOW = "FFFF00"
    BLUE = "0000FF"

    # highlight rules const
    OVERFLOW_LIST = ['nan\t', 'inf\t', '-inf\t', 'nan', 'inf', '-inf']
    MAX_DIFF_RED = 1e+10
    ORDER_MAGNITUDE_DIFF_YELLOW = 1
    ONE_THOUSAND_ERROR_IN_RED = 0.9
    ONE_THOUSAND_ERROR_OUT_RED = 0.6
    ONE_THOUSAND_ERROR_DIFF_YELLOW = 0.1
    COSINE_DIFF_YELLOW = 0.1
    MAX_RELATIVE_OUT_RED = 0.5
    MAX_RELATIVE_OUT_YELLOW = 0.1
    MAX_RELATIVE_IN_YELLOW = 0.01


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


    def __init__(self, code, error_info: str = ""):
        super(CompareException, self).__init__()
        self.code = code
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class DumpException(CompareException):
    pass


class OverflowConst:
    """
    Class for Overflow
    """
    OVERFLOW_DEBUG_MODE_ENABLE = "OVERFLOW_DEBUG_MODE_ENABLE"
    OVERFLOW_ORIGINAL_MODE = 0
    OVERFLOW_DEBUG_MODE = 1


def make_dump_path_if_not_exists(dump_path):
    if not os.path.exists(dump_path):
        try:
            Path(dump_path).mkdir(mode=0o750, exist_ok=True, parents=True)
        except OSError as ex:
            print_error_log(
                'Failed to create {}.Please check the path permission or disk space .{}'.format(dump_path, str(ex)))
            raise CompareException(CompareException.INVALID_PATH_ERROR) from ex
    else:
        if not os.path.isdir(dump_path):
            print_error_log('{} already exists and is not a directory.'.format(dump_path))


def _print_log(level, msg, end='\n'):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getgid()
    print(current_time + "(" + str(pid) + ")-[" + level + "]" + msg, end=end)
    sys.stdout.flush()


def print_info_log(info_msg, end='\n'):
    """
    Function Description:
        print info log.
    Parameter:
        info_msg: the info message.
    """
    _print_log("INFO", info_msg, end=end)


def print_error_log(error_msg):
    """
    Function Description:
        print error log.
    Parameter:
        error_msg: the error message.
    """
    _print_log("ERROR", error_msg)


def print_warn_log(warn_msg):
    """
    Function Description:
        print warn log.
    Parameter:
        warn_msg: the warning message.
    """
    _print_log("WARNING", warn_msg)


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
        print_error_log("Please set switch with 'ON' or 'OFF'.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)


def check_dump_mode_valid(dump_mode):
    if not isinstance(dump_mode, list):
        print_warn_log("Please set dump_mode as a list.")
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
        print_error_log("Params summary_only only support True or False.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    return summary_only


def check_compare_param(input_parma, output_path, stack_mode=False, summary_compare=False, md5_compare=False):
    if not (isinstance(input_parma, dict) and isinstance(output_path, str)):
        print_error_log("Invalid input parameters")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    check_file_or_directory_path(input_parma.get("npu_json_path"), False)
    check_file_or_directory_path(input_parma.get("bench_json_path"), False)
    check_file_or_directory_path(input_parma.get("stack_json_path"), False)
    if not summary_compare and not md5_compare:
        check_file_or_directory_path(input_parma.get("npu_dump_data_dir"), True)
        check_file_or_directory_path(input_parma.get("bench_dump_data_dir"), True)
    check_file_or_directory_path(output_path, True)
    with FileOpen(input_parma.get("npu_json_path"), "r") as npu_json, \
         FileOpen(input_parma.get("bench_json_path"), "r") as bench_json, \
         FileOpen(input_parma.get("stack_json_path"), "r") as stack_json:
        check_json_file(input_parma, npu_json, bench_json, stack_json)


def check_configuration_param(stack_mode=False, auto_analyze=True, fuzzy_match=False):
    if not (isinstance(stack_mode, bool) and isinstance(auto_analyze, bool) and isinstance(fuzzy_match, bool)):
        print_error_log("Invalid input parameters which should be only bool type.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)


def check_file_or_directory_path(path, isdir=False):
    """
    Function Description:
        check whether the path is valid
    Parameter:
        path: the path to check
        isdir: the path is dir or file
    Exception Description:
        when invalid data throw exception
    """
    if isdir:
        path_checker = FileChecker(path, FileCheckConst.DIR, FileCheckConst.WRITE_ABLE)
    else:
        path_checker = FileChecker(path, FileCheckConst.FILE, FileCheckConst.READ_ABLE)
    path_checker.common_check()


def is_starts_with(string, prefix_list):
    return any(string.startswith(prefix) for prefix in prefix_list)


def _check_json(json_file_handle, file_name):
    tensor_line = json_file_handle.readline()
    if not tensor_line:
        print_error_log("dump file {} have empty line!".format(file_name))
        raise CompareException(CompareException.INVALID_DUMP_FILE)
    json_file_handle.seek(0, 0)


def check_json_file(input_param, npu_json, bench_json, stack_json):
    _check_json(npu_json, input_param.get("npu_json_path"))
    _check_json(bench_json, input_param.get("bench_json_path"))
    _check_json(stack_json, input_param.get("stack_json_path"))


def check_file_size(input_file, max_size):
    try:
        file_size = os.path.getsize(input_file)
    except OSError as os_error:
        print_error_log('Failed to open "%s". %s' % (input_file, str(os_error)))
        raise CompareException(CompareException.INVALID_FILE_ERROR) from os_error
    if file_size > max_size:
        print_error_log('The size (%d) of %s exceeds (%d) bytes, tools not support.'
                        % (file_size, input_file, max_size))
        raise CompareException(CompareException.INVALID_FILE_ERROR)


def check_file_not_exists(file_path):
    if os.path.exists(file_path) or os.path.islink(file_path):
        remove_path(file_path)


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


def remove_path(path):
    if not os.path.exists(path):
        return
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    except PermissionError as err:
        print_error_log("Failed to delete {}. Please check the permission.".format(path))
        raise CompareException(CompareException.INVALID_PATH_ERROR) from err


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
    for dir_path, sub_paths, files in os.walk(dump_dir):
        if len(files) != 0:
            dump_data_path = dir_path
            file_is_exist = True
            break
        dump_data_path = dir_path
    return dump_data_path, file_is_exist


def modify_dump_path(dump_path, mode):
    if mode == Const.ALL:
        return dump_path
    file_name = os.path.split(dump_path)
    mode_file_name = mode + "_" + file_name[-1]
    return os.path.join(file_name[0], mode_file_name)


def create_directory(dir_path):
    """
    Function Description:
        creating a directory with specified permissions
    Parameter:
        dir_path: directory path
    Exception Description:
        when invalid data throw exception
    """
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, mode=0o700)
        except OSError as ex:
            print_error_log(
                'Failed to create {}.Please check the path permission or disk space .{}'.format(dir_path, str(ex)))
            raise CompareException(CompareException.INVALID_PATH_ERROR) from ex


def execute_command(cmd):
    """
    Function Description:
        run the following command
    Parameter:
        cmd: command
    Exception Description:
        when invalid command throw exception
    """
    print_info_log('Execute command:%s' % cmd)
    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while process.poll() is None:
        line = process.stdout.readline()
        line = line.strip()
        if line:
            print(line)
    if process.returncode != 0:
        print_error_log('Failed to execute command:%s' % " ".join(cmd))
        raise CompareException(CompareException.INVALID_DATA_ERROR)


def save_numpy_data(file_path, data):
    """
    save_numpy_data
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    np.save(file_path, data)


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
            print_error_log("please check your input shape.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
    return value_list


def get_data_len_by_shape(shape):
    data_len = 1
    for item in shape:
        if item == -1:
            print_error_log("please check your input shape, one dim in shape is -1.")
            return -1
        data_len = data_len * item
    return data_len


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
            print_error_log(f"Seed must be between 0 and {Const.MAX_SEED_VALUE}.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
    else:
        print_error_log(f"Seed must be integer.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    if not isinstance(mode, bool):
        print_error_log(f"seed_all mode must be bool.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)


def get_process_rank(model):
    print_info_log("Rank id is not provided. Trying to get the rank id of the model.")
    try:
        local_device = next(model.parameters()).device
    except StopIteration:
        print_warn_log('There is no parameter in the model. Fail to get rank id.')
        return 0, False
    if local_device.type == 'cpu':
        print_warn_log("Warning: the debugger is unable to get the rank id. "
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
        print_error_log(f"Failed to open file. Please check file {template_path} or path {pkl_dir}.")

    print_info_log(f"Generate compare script successfully which is {compare_script_path}.")


def check_file_valid(file_path):
    if os.path.islink(file_path):
        print_error_log('The file path {} is a soft link.'.format(file_path))
        raise CompareException(CompareException.INVALID_PATH_ERROR)

    if len(os.path.realpath(file_path)) > Const.DIRECTORY_LENGTH or len(os.path.basename(file_path)) > \
            Const.FILE_NAME_LENGTH:
        print_error_log('The file path length exceeds limit.')
        raise CompareException(CompareException.INVALID_PATH_ERROR)

    if not re.match(Const.FILE_PATTERN, os.path.realpath(file_path)):
        print_error_log('The file path {} contains special characters.'.format(file_path))
        raise CompareException(CompareException.INVALID_PATH_ERROR)

    if os.path.isfile(file_path):
        file_size = os.path.getsize(file_path)
        if file_path.endswith(Const.PKL_SUFFIX) and file_size > Const.ONE_GB:
            print_error_log('The file {} size is greater than 1GB.'.format(file_path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)
        if file_path.endswith(Const.NUMPY_SUFFIX) and file_size > Const.TEN_GB:
            print_error_log('The file {} size is greater than 10GB.'.format(file_path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)


def check_path_before_create(path):
    if len(os.path.realpath(path)) > Const.DIRECTORY_LENGTH or len(os.path.basename(path)) > \
            Const.FILE_NAME_LENGTH:
        print_error_log('The file path length exceeds limit.')
        raise CompareException(CompareException.INVALID_PATH_ERROR)

    if not re.match(Const.FILE_PATTERN, os.path.realpath(path)):
        print_error_log('The file path {} contains special characters.'.format(path))
        raise CompareException(CompareException.INVALID_PATH_ERROR)


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
    npu_json_path = input_param.get("npu_json_path", None)
    bench_json_path = input_param.get("bench_json_path", None)
    if not npu_json_path or not bench_json_path:
        print_error_log(f"Please check the json path is valid.")
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    with FileOpen(npu_json_path, 'r') as npu_f:
        npu_json_data = json.load(npu_f)
    with FileOpen(bench_json_path, 'r') as bench_f:
        bench_json_data = json.load(bench_f)
    if npu_json_data['task'] != bench_json_data['task']:
        print_error_log(f"Please check the dump task is consistent.")
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
        print_error_log(f"Compare is not required for overflow_check or free_benchmark.")
        raise CompareException(CompareException.INVALID_TASK_ERROR)
    input_param['npu_dump_data_dir'] = npu_json_data['dump_data_dir']
    input_param['bench_dump_data_dir'] = bench_json_data['dump_data_dir']
    return summary_compare, md5_compare


def get_header_index(header_name, summary_compare=False):
    if summary_compare:
        header = CompareConst.SUMMARY_COMPARE_RESULT_HEADER[:]
    else:
        header = CompareConst.COMPARE_RESULT_HEADER[:]
    if header_name not in header:
        print_error_log(f"{header_name} not in data name")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    return header.index(header_name)
