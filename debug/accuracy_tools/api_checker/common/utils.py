import os
import json
import csv
import sys
import time
import stat
import inspect

import numpy as np
import torch

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
    
    INPUT = "input"
    OUTPUT = "output"

    # dump mode
    ALL = "all"
    LIST = "list"
    RANGE = "range"
    STACK = "stack"
    ACL = "acl"
    API_LIST = "api_list"
    API_STACK = "api_stack"
    DUMP_MODE = [ALL, LIST, RANGE, STACK, ACL, API_LIST, API_STACK]

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


def get_json_contents(file_path):
    with open(file_path, "r") as f:
        ops = f.read()
    #ops = get_file_content_bytes(file_path)
    try:
        json_obj = json.loads(ops)
    except ValueError as error:
        print_error_log('Failed to load "%s". %s' % (file_path, str(error)))
        #raise CompareException(CompareException.INVALID_FILE_ERROR) from error
    if not isinstance(json_obj, dict):
        print_error_log('Json file %s, content is not a dictionary!' % file_path)
        #raise CompareException(CompareException.INVALID_FILE_ERROR)
    return json_obj


def write_csv(data, filepath):
    with open(filepath, 'a', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(data)


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
            #raise CompareException(CompareException.INVALID_PATH_ERROR)

        if not os.path.isdir(path):
            print_error_log('The path {} is not a directory.'.format(path))
            #raise CompareException(CompareException.INVALID_PATH_ERROR)

        if not os.access(path, os.W_OK):
            print_error_log(
                'The path {} does not have permission to write. Please check the path permission'.format(path))
            #raise CompareException(CompareException.INVALID_PATH_ERROR)
    else:
        if not os.path.isfile(path):
            print_error_log('{} is an invalid file or non-exist.'.format(path))
            #raise CompareException(CompareException.INVALID_PATH_ERROR)

    if not os.access(path, os.R_OK):
        print_error_log(
            'The path {} does not have permission to read. Please check the path permission'.format(path))
        #raise CompareException(CompareException.INVALID_PATH_ERROR)


def get_stack():
    stack_str = []
    try:
        for (_, path, line, func, code, _) in inspect.stack()[3:]:
            if code:
                stack_line = [path, str(line), func, code[0].strip() if code else code]
            else:
                stack_line = [path, str(line), func, code]
            stack_str.append(stack_line)
    except Exception as e:
        print("Dump stack info failed, error: {}".format(e))
        stack_str.append('') 
    return stack_str


dtype_map = {
    "Float32": np.float32,
    "Float16": np.float16,
    "Float64": np.float64,
    "Int8": np.int8,
    "Int16": np.int16,
    "Int32": np.int32,
    "Int64": np.int64,
    "Bool_": np.bool_,
    "Uint8": np.uint8,
    "Uint16": np.uint16,
    "Uint32": np.uint32,
    "Uint64": np.uint64,
    "Bool": np.bool_,
    "Complex64": np.complex64,
    "Complex128": np.complex128
}


np_scalar_type = [
    bool,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]