#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2023-2023. Huawei Technologies Co., Ltd. All rights reserved.
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
import json
import os
import random
import re
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import csv

try:
    import torch_npu
except ImportError:
    IS_GPU = True
else:
    IS_GPU = False

from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileCheckConst, FileChecker, FileOpen
from ptdbg_ascend.src.python.ptdbg_ascend.common import file_check_util

torch_without_guard_version_list = ['2.1']
for version in torch_without_guard_version_list:
    if torch.__version__.startswith(version):
        torch_without_guard_version = True
        break
    else:
        torch_without_guard_version = False
if not IS_GPU and not torch_without_guard_version:
    from torch_npu.utils.device_guard import torch_device_guard as torch_npu_device_guard


class Const:
    """
    Class for const
    """
    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    MODEL_TYPE = ['.onnx', '.pb', '.om']
    SEMICOLON = ";"
    COLON = ":"
    EQUAL = "="
    COMMA = ","
    DOT = "."
    DUMP_RATIO_MAX = 100
    SUMMERY_DATA_NUMS = 256
    ONE_HUNDRED_MB = 100 * 1024 * 1024
    FLOAT_EPSILON = np.finfo(float).eps
    SUPPORT_DUMP_MODE = ['api', 'acl']
    ON = 'ON'
    OFF = 'OFF'
    BACKWARD = 'backward'
    FORWARD = 'forward'
    FLOAT_TYPE = [np.half, np.single, float, np.double, np.float64, np.longdouble, np.float32, np.float16]
    BOOL_TYPE = [bool, np.uint8]
    INT_TYPE = [np.int32, np.int64]

    # dump mode
    ALL = "all"
    LIST = "list"
    RANGE = "range"
    STACK = "stack"
    ACL = "acl"
    API_LIST = "api_list"
    API_STACK = "api_stack"
    DUMP_MODE = [ALL, LIST, RANGE, STACK, ACL, API_LIST, API_STACK]

    API_PATTERN = r"^[A-Za-z0-9]+[_]+([A-Za-z0-9]+[_]*[A-Za-z0-9]+)[_]+[0-9]+[_]+[A-Za-z0-9]+"
    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR

    RAISE_PRECISION = {
        torch.float16: torch.float32,
        torch.bfloat16: torch.float32,
        torch.float32: torch.float64
    }
    CONVERT = {
        "int32_to_int64": ["torch.int32", "torch.int64"],
    }

    CONVERT_API = {
        "int32_to_int64": ["cross_entropy"]
    }


class CompareConst:
    """
    Class for compare module const
    """
    # compare result column name
    NPU_NAME = "NPU Name"
    BENCH_NAME = "Bench Name"
    NPU_DTYPE = "NPU Tensor Dtype"
    BENCH_DTYPE = "Bench Tensor Dtype"
    NPU_SHAPE = "NPU Tensor Shape"
    BENCH_SHAPE = "Bench Tensor Shape"
    NPU_MAX = "NPU max"
    NPU_MIN = "NPU min"
    NPU_MEAN = "NPU mean"
    BENCH_MAX = "Bench max"
    BENCH_MIN = "Bench min"
    BENCH_MEAN = "Bench mean"
    COSINE = "Cosine"
    MAX_ABS_ERR = "MaxAbsErr"
    ACCURACY = "Accuracy Reached or Not"
    STACK = "NPU_Stack_Info"
    ERROR_MESSAGE = "Err_message"

    # compare result data
    NAN = 'Nan'
    SHAPE_UNMATCH = 'shape unmatched'
    DTYPE_UNMATCH = 'dtype unmatched'

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


class VersionCheck:
    """
    Class for TorchVersion
    """
    V1_8 = "1.8"
    V1_11 = "1.11"

    @staticmethod
    def check_torch_version(version):
        torch_version = torch.__version__
        if torch_version.startswith(version):
            return True
        else:
            return False


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

    def __init__(self, code, error_info: str = ""):
        super(CompareException, self).__init__()
        self.code = code
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class DumpException(CompareException):
    pass


def read_json(file):
    with FileOpen(file, 'r') as f:
        obj = json.load(f)
    return obj


def write_csv(data, filepath):
    with FileOpen(filepath, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def _print_log(level, msg):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getgid()
    print(current_time + "(" + str(pid) + ")-[" + level + "]" + msg)
    sys.stdout.flush()


def print_info_log(info_msg):
    """
    Function Description:
        print info log.
    Parameter:
        info_msg: the info message.
    """
    _print_log("INFO", info_msg)


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


def check_mode_valid(mode):
    if mode not in Const.DUMP_MODE:
        msg = "Current mode '%s' is not supported. Please use the field in %s" % \
              (mode, Const.DUMP_MODE)
        raise CompareException(CompareException.INVALID_DUMP_MODE, msg)


def check_object_type(check_object, allow_type):
    """
    Function Description:
        Check if the object belongs to a certain data type
    Parameter:
        check_object: the object to be checked
        allow_type: legal data type
    Exception Description:
        when invalid data throw exception
    """
    if not isinstance(check_object, allow_type):
        print_error_log(f"{check_object} not of {allow_type} type")
        raise CompareException(CompareException.INVALID_DATA_ERROR)


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
        if not os.path.exists(path):
            print_error_log('The path {} is not exist.'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

        if not os.path.isdir(path):
            print_error_log('The path {} is not a directory.'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

        if not os.access(path, os.W_OK):
            print_error_log(
                'The path {} does not have permission to write. Please check the path permission'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)
    else:
        if not os.path.isfile(path):
            print_error_log('{} is an invalid file or non-exist.'.format(path))
            raise CompareException(CompareException.INVALID_PATH_ERROR)

    if not os.access(path, os.R_OK):
        print_error_log(
            'The path {} does not have permission to read. Please check the path permission'.format(path))
        raise CompareException(CompareException.INVALID_PATH_ERROR)


def _check_pkl(pkl_file_handle, file_name):
    tensor_line = pkl_file_handle.readline()
    if len(tensor_line) == 0:
        print_error_log("dump file {} have empty line!".format(file_name))
        raise CompareException(CompareException.INVALID_DUMP_FILE)
    pkl_file_handle.seek(0, 0)


def check_file_mode(npu_pkl, bench_pkl, stack_mode):
    npu_pkl_name = os.path.split(npu_pkl)[-1]
    bench_pkl_name = os.path.split(bench_pkl)[-1]

    if not npu_pkl_name.startswith("api_stack") and not bench_pkl_name.startswith("api_stack"):
        if stack_mode:
            print_error_log("The current file does not contain stack information, please turn off the stack_mode")
            raise CompareException(CompareException.INVALID_COMPARE_MODE)
    elif npu_pkl_name.startswith("api_stack") and bench_pkl_name.startswith("api_stack"):
        if not stack_mode:
            print_error_log("The current file contains stack information, please turn on the stack_mode")
            raise CompareException(CompareException.INVALID_COMPARE_MODE)
    else:
        print_error_log("The dump mode of the two files is not same, please check the dump files")
        raise CompareException(CompareException.INVALID_COMPARE_MODE)


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


def get_api_name_from_matcher(name):
    api_matcher = re.compile(Const.API_PATTERN)
    match = api_matcher.match(name)
    return match.group(1) if match else ""


def modify_dump_path(dump_path, mode):
    if mode == Const.ALL:
        return dump_path
    file_name = os.path.split(dump_path)
    mode_file_name = mode + "_" + file_name[-1]
    return os.path.join(file_name[0], mode_file_name)


def create_directory(dir_path):
    """
    Function Description:
        creating a directory with specified permissions in a thread-safe manner
    Parameter:
        dir_path: directory path
    Exception Description:
        when invalid data throw exception
    """
    try:
        os.makedirs(dir_path, mode=FileCheckConst.DATA_DIR_AUTHORITY, exist_ok=True)
    except OSError as ex:
        print_error_log(
            'Failed to create {}. Please check the path permission or disk space. {}'.format(dir_path, str(ex)))
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


def parse_arg_value(values):
    """
    parse dynamic arg value of atc cmdline
    """
    value_list = []
    for item in values.split(Const.SEMICOLON):
        value_list.append(parse_value_by_comma(item))
    return value_list


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


def get_time():
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def format_value(value):
    return '{:.6f}'.format(value)


def torch_device_guard(func):
    if IS_GPU or torch_without_guard_version:
        return func
    # Parse args/kwargs matched torch.device objects

    @torch_npu_device_guard
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def seed_all(seed=1234, mode=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)
    if IS_GPU:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.benchmark = False
    else:
        torch_npu.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)


def get_process_rank(model):
    print_info_log("Rank id is not provided. Trying to get the rank id of the model.")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        print_warn_log('There is no parameter in the model. Fail to get rank id.')
        return 0, False
    if device.type == 'cpu':
        print_warn_log("Warning: the debugger is unable to get the rank id. "
            "This may cause the dumpped data to be corrupted in the "
            "case of distributed training. (You may ignore this if you are using only one card.) "
            "Transfer the model to npu or gpu before register_hook() to avoid this warning.")
        return 0, False
    else:
        return device.index, True


def get_json_contents(file_path):
    ops = get_file_content_bytes(file_path)
    try:
        json_obj = json.loads(ops)
    except ValueError as error:
        print_error_log('Failed to load "%s". %s' % (file_path, str(error)))
        raise CompareException(CompareException.INVALID_FILE_ERROR) from error
    if not isinstance(json_obj, dict):
        print_error_log('Json file %s, content is not a dictionary!' % file_path)
        raise CompareException(CompareException.INVALID_FILE_ERROR)
    return json_obj


def get_file_content_bytes(file):
    with FileOpen(file, 'rb') as file_handle:
        return file_handle.read()


def islink(path):
    path = os.path.abspath(path)
    return os.path.islink(path)


class SoftlinkCheckException(Exception):
    pass


MAX_JSON_FILE_SIZE = 10 * 1024 ** 2
LINUX_FILE_NAME_LENGTH_LIMIT = 200


def check_path_length_valid(path):
    path = os.path.realpath(path)
    return len(os.path.basename(path)) <= LINUX_FILE_NAME_LENGTH_LIMIT


def check_path_pattern_valid(path):
    pattern = re.compile(r'(\.|/|:|_|-|\s|[~0-9a-zA-Z])+')
    if not pattern.fullmatch(path):
        raise ValueError('Only the following characters are allowed in the path: A-Z a-z 0-9 - _ . / :')


def check_input_file_valid(input_path, max_file_size=MAX_JSON_FILE_SIZE):
    if islink(input_path):
        raise SoftlinkCheckException("Input path doesn't support soft link.")

    input_path = os.path.realpath(input_path)
    if not os.path.exists(input_path):
        raise ValueError('Input file %s does not exist!' % input_path)

    if not os.access(input_path, os.R_OK):
        raise PermissionError('Input file %s is not readable!' % input_path)

    if not check_path_length_valid(input_path):
        raise ValueError("The real path or file_name of input is too long.")

    check_path_pattern_valid(input_path)

    if os.path.getsize(input_path) > max_file_size:
        raise ValueError(f'The file is too large, exceeds {max_file_size // 1024 ** 2}MB')


def check_need_convert(api_name):
    convert_type = None
    for key, value in Const.CONVERT_API.items():
        if api_name not in value:
            continue
        else:
            convert_type = key
    return convert_type


def api_info_preprocess(api_name, api_info_dict):
    """
    Function Description:
        Preprocesses the API information.
    Parameter:
        api_name: Name of the API.
        api_info_dict: argument of the API.
    Return api_info_dict:
        convert_type: Type of conversion.
        api_info_dict: Processed argument of the API.
    """
    convert_type = check_need_convert(api_name)
    if api_name == 'cross_entropy':
        api_info_dict = cross_entropy_process(api_info_dict)
    return convert_type, api_info_dict


def cross_entropy_process(api_info_dict):
    """
    Function Description:
        Preprocesses the cross_entropy API information.
    Parameter:
        api_info_dict: argument of the API.
    Return api_info_dict:
        api_info_dict: Processed argument of the API.
    """
    if 'args' in api_info_dict and len(api_info_dict['args']) > 1 and 'Min' in api_info_dict['args'][1]:
        if api_info_dict['args'][1]['Min'] <= 0:
            api_info_dict['args'][1]['Min'] = 0 #The second argument in cross_entropy should be -100 or not less than 0.
    return api_info_dict


def initialize_save_path(save_path, dir_name):
    data_path = os.path.join(save_path, dir_name)
    if os.path.exists(data_path):
        print_warn_log(f"{data_path} already exists, it will be overwritten")
    else:
        os.mkdir(data_path, mode=FileCheckConst.DATA_DIR_AUTHORITY)
    data_path_checker = FileChecker(data_path, FileCheckConst.DIR)
    data_path_checker.common_check()


def write_pt(file_path, tensor):
    if os.path.exists(file_path):
        raise ValueError(f"File {file_path} already exists")
    torch.save(tensor, file_path)
    full_path = os.path.realpath(file_path)
    file_check_util.change_mode(full_path, FileCheckConst.DATA_FILE_AUTHORITY)
    return full_path
