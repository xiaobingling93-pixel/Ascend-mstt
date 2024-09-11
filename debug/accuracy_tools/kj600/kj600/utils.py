import os
import time
import sys
import re
from functools import wraps
from torch import distributed as dist

from kj600.const import Const

FILE_MAX_SIZE = 10 * 1024 * 1024 * 1024
FILE_NAME_MAX_LENGTH = 255
DIRECTORY_MAX_LENGTH = 4096
FILE_NAME_VALID_PATTERN = r"^[a-zA-Z0-9_.:/-]+$"



class MsgConst:
    """
    Class for log messages const
    """
    SPECIAL_CHAR = ["\n", "\r", "\u007F", "\b", "\f", "\t", "\u000B", "%08", "%0a", "%0b", "%0c", "%0d", "%7f"]


def filter_special_chars(func):
    @wraps(func)
    def func_level(msg):
        for char in MsgConst.SPECIAL_CHAR:
            msg = msg.replace(char, '_')
        return func(msg)
    return func_level


class FileCheckConst:
    """
    Class for file check const
    """
    READ_ABLE = "read"
    WRITE_ABLE = "write"
    READ_WRITE_ABLE = "read and write"
    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FILE_VALID_PATTERN = r"^[a-zA-Z0-9_.:/-]+$"
    PKL_SUFFIX = ".pkl"
    NUMPY_SUFFIX = ".npy"
    JSON_SUFFIX = ".json"
    PT_SUFFIX = ".pt"
    CSV_SUFFIX = ".csv"
    YAML_SUFFIX = ".yaml"
    MAX_PKL_SIZE = 1 * 1024 * 1024 * 1024
    MAX_NUMPY_SIZE = 10 * 1024 * 1024 * 1024
    MAX_JSON_SIZE = 1 * 1024 * 1024 * 1024
    MAX_PT_SIZE = 10 * 1024 * 1024 * 1024
    MAX_CSV_SIZE = 1 * 1024 * 1024 * 1024
    MAX_YAML_SIZE = 10 * 1024 * 1024
    DIR = "dir"
    FILE = "file"
    DATA_DIR_AUTHORITY = 0o750
    DATA_FILE_AUTHORITY = 0o640
    FILE_SIZE_DICT = {
        PKL_SUFFIX: MAX_PKL_SIZE,
        NUMPY_SUFFIX: MAX_NUMPY_SIZE,
        JSON_SUFFIX: MAX_JSON_SIZE,
        PT_SUFFIX: MAX_PT_SIZE,
        CSV_SUFFIX: MAX_CSV_SIZE,
        YAML_SUFFIX: MAX_YAML_SIZE
    }

class FileCheckException(Exception):
    """
    Class for File Check Exception
    """
    NONE_ERROR = 0
    INVALID_PATH_ERROR = 1
    INVALID_FILE_TYPE_ERROR = 2
    INVALID_PARAM_ERROR = 3
    INVALID_PERMISSION_ERROR = 3

    def __init__(self, code, error_info: str = ""):
        super(FileCheckException, self).__init__()
        self.code = code
        self.error_info = error_info

    def __str__(self):
        return self.error_info


def print_rank_0(message):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message)
    else:
        print(message)


def _print_log(level, msg, end='\n'):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getgid()
    print(current_time + "(" + str(pid) + ")-[" + level + "]" + msg, end=end)
    sys.stdout.flush()


@filter_special_chars
def print_info_log(info_msg):
    """
    Function Description:
        print info log.
    Parameter:
        info_msg: the info message.
    """
    _print_log("INFO", info_msg)


@filter_special_chars
def print_error_log(error_msg):
    """
    Function Description:
        print error log.
    Parameter:
        error_msg: the error message.
    """
    _print_log("ERROR", error_msg)


@filter_special_chars
def print_warn_log(warn_msg):
    """
    Function Description:
        print warn log.
    Parameter:
        warn_msg: the warning message.
    """
    _print_log("WARNING", warn_msg)

def get_param_struct(param):
    if isinstance(param, tuple):
        return f"tuple[{len(param)}]"
    if isinstance(param, list):
        return f"list[{len(param)}]"
    return "tensor"

def check_link(path):
    abs_path = os.path.abspath(path)
    if os.path.islink(abs_path):
        raise RuntimeError("The path is a soft link.")


def check_path_length(path, name_length_limit=None):
    file_max_name_length = name_length_limit if name_length_limit else FILE_NAME_MAX_LENGTH
    if len(path) > DIRECTORY_MAX_LENGTH or \
            len(os.path.basename(path)) > file_max_name_length:
        raise RuntimeError("The file path length exceeds limit.")


def check_path_pattern_valid(path):
    if not re.match(FILE_NAME_VALID_PATTERN, path):
        raise RuntimeError("The file path contains special characters.")


def check_path_readability(path):
    if not os.access(path, os.R_OK):
        raise RuntimeError("The file path is not readable.")


def check_path_writability(path):
    if not os.access(path, os.W_OK):
        raise RuntimeError("The file path is not writable.")


def check_file_size(file_path, max_size=FILE_MAX_SIZE):
    file_size = os.path.getsize(file_path)
    if file_size >= max_size:
        raise RuntimeError("The file size excess limit.")


def check_path_exists(path):
    if not os.path.exists(path):
        raise RuntimeError("The file path does not exist.")


def check_file_valid(path):
    check_path_exists(path)
    check_link(path)
    real_path = os.path.realpath(path)
    check_path_length(real_path)
    check_path_pattern_valid(real_path)
    check_file_size(real_path)


def check_file_valid_readable(path):
    check_file_valid(path)
    check_path_readability(path)


def check_file_valid_writable(path):
    check_file_valid(path)
    check_path_writability(path)


def change_mode(path, mode):
    if not os.path.exists(path) or os.path.islink(path):
        return
    try:
        os.chmod(path, mode)
    except PermissionError as ex:
        print_error_log('Failed to change {} authority. {}'.format(path, str(ex)))
        raise FileCheckException(FileCheckException.INVALID_PERMISSION_ERROR) from ex

def validate_ops(ops):
    if not isinstance(ops, list):
        raise Exception("ops should be a list")
    if not ops:
        raise Exception(f"specify ops to calculate metrics. Optional ops: {Const.OP_LIST}")

    valid_ops = []
    for op in ops:
        if op not in Const.OP_LIST:
            raise Exception(f"op {op} is not supported. Optional ops: {Const.OP_LIST}")
        else:
            valid_ops.append(op)
    return valid_ops

def validate_ranks(ranks):
    world_size = dist.get_world_size()
    if not isinstance(ranks, list):
        raise Exception("module_ranks should be a list")
    for rank in ranks:
        if not isinstance(rank, int):
            raise Exception("element in module_ranks should be a int, get {type(rank)}")
        if rank < 0 or rank >= world_size:
            print_warn_log(f"rank {rank} should be in rang [0, {world_size}]")

def validate_config(config):
    config['ops'] = validate_ops(config.get('ops', []))
    ranks = config.get("module_ranks", [])
    validate_ranks(ranks)

    targets = config.get("targets", {})
    if not isinstance(targets, dict):
        raise ValueError('targets in config.json should be a dict')
    