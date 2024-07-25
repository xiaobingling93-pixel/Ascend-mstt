import os
import re
import sys
import time
import yaml

import pandas as pd

from grad_tool.common.constant import GradConst


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


def write_csv(filepath, content_list, header):
    if not os.path.exists(filepath):
        make_file_safety(filepath)
        data_frame = pd.DataFrame(columns=header)
        data_frame.to_csv(filepath, index=False)

    check_file_or_directory_path(filepath)
    new_data = pd.DataFrame(list(content for content in content_list))
    new_data.to_csv(filepath, mode='a+', header=False, index=False)


def make_file_safety(file_path: str, permission=0o640):
    if os.path.islink(file_path):
        raise RuntimeError(f"Invalid soft link path: {file_path}")
    file_real_path = os.path.realpath(file_path)
    if os.path.exists(file_real_path):
        return
    parent_path = os.path.dirname(file_real_path)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path, mode=GradConst.DATA_DIR_AUTHORITY, exist_ok=True)
    if not os.access(parent_path, os.W_OK):
        raise PermissionError(f"The path {parent_path} is not writable!")
    try:
        os.close(os.open(file_real_path, os.O_WRONLY | os.O_CREAT, permission))
    except OSError as e:
        raise RuntimeError("Can't create file: " + file_real_path) from e
    os.chmod(file_real_path, permission)


def data_in_list_target(data, lst):
    return not lst or len(lst) == 0 or data in lst


def check_numeral_list_ascend(lst):
    if any(not isinstance(item, (int, float)) for item in lst):
        raise Exception("The input list should only contain numbers")
    if lst != sorted(lst):
        raise Exception("The input list should be ascending")


class ListCache(list):
    threshold = 1000

    def __init__(self, *args):
        super().__init__(*args)

    def __del__(self):
        self.flush()

    def flush(self):
        if len(self) == 0:
            return
        if not self._output_file:
            print_warn_log("dumpfile path is not setted")
        write_csv(self._output_file, self, [])
        print_info_log(f"write {len(self)} items to {self._output_file}.")
        self.clear()

    def append(self, data):
        list.append(self, data)
        if len(self) >= ListCache.threshold:
            self.flush()
    
    def set_output_file(self, output_file):
        self._output_file = output_file


def get_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config


def check_link(path):
    abs_path = os.path.abspath(path)
    if os.path.islink(abs_path):
        raise RuntimeError("The path is a soft link.")


def check_path_length(path, name_length=None):
    file_max_name_length = name_length if name_length else GradConst.FILE_NAME_LENGTH
    if len(path) > GradConst.DIRECTORY_LENGTH or \
            len(os.path.basename(path)) > file_max_name_length:
        raise RuntimeError("The file path length exceeds limit.")


def check_path_pattern_vaild(path):
    if not re.match(GradConst.FILE_VALID_PATTERN, path):
        raise RuntimeError("The file path contains special characters.")


def check_path_readability(path):
    if not os.access(path, os.R_OK):
        raise RuntimeError("The file path is not readable.")


def check_path_writability(path):
    if not os.access(path, os.W_OK):
        raise RuntimeError("The file path is not writable.")


def check_path_owner_consistent(path):
    file_owner = os.stat(path).st_uid
    if file_owner != os.getuid():
        raise RuntimeError("The file path may be insecure because is does not belong to you.")


def _user_interactive_confirm(message):
    while True:
        check_message = input(message + " Enter 'c' to continue or enter 'e' to exit: ")
        if check_message == "c":
            break
        elif check_message == "e":
            print_warn_log("User canceled.")
            raise RuntimeError("User canceled.")
        else:
            print("Input is error, please enter 'c' or 'e'.")


def check_file_size(file_path, max_size=GradConst.MAX_SIZE):
    file_size = os.path.getsize(file_path)
    if file_size >= max_size:
        _user_interactive_confirm(f'The size of file path {file_path} exceeds {max_size} bytes.'
                                  f'Do you want to continue?')


def check_path_type(file_path, file_type):
    if file_type == GradConst.FILE:
        if not os.path.isfile(file_path):
            raise RuntimeError("The path should be a file!")
    if file_type == GradConst.DIR:
        if not os.path.isdir(file_path):
            raise RuntimeError("The path should be a dictionary!")


def check_path_exists(path):
    if not os.path.exists(path):
        raise RuntimeError("The file path does not exist.")


def path_valid_check(path):
    check_path_length(path)
    check_path_pattern_vaild(path)


def check_file_or_directory_path(path, file_type=GradConst.FILE):
    check_link(path)
    check_path_exists(path)
    check_path_length(path)
    check_path_pattern_vaild(path)
    check_path_owner_consistent(path)
    check_path_type(path, file_type)
    if file_type == GradConst.FILE:
        check_path_readability(path)
        check_file_size(path)
    else:
        check_path_writability(path)


def create_directory(dir_path):
    dir_path = os.path.realpath(dir_path)
    try:
        os.makedirs(dir_path, mode=GradConst.DATA_DIR_AUTHORITY, exist_ok=True)
    except OSError as ex:
        raise RuntimeError("Failed to create directory. Please check the path permission or disk space.") from ex

def change_mode(path, mode):
    check_path_exists(path)
    check_link(path)
    try:
        os.chmod(path, mode)
    except PermissionError as ex:
        print_error_log(f'Failed to change {path} authority. {str(ex)}')
        raise ex

def check_param(param_name):
    if not re.match(GradConst.PARAM_VALID_PATTERN, param_name):
        raise RuntimeError("The parameter name contains special characters.")