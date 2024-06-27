import os
import time
import sys
import re

FILE_MAX_SIZE = 10 * 1024 * 1024 * 1024
FILE_NAME_MAX_LENGTH = 255
DIRECTORY_MAX_LENGTH = 4096
FILE_NAME_VALID_PATTERN = r"^[a-zA-Z0-9_.:/-]+$"

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


def check_path_pattern_vaild(path):
    if not re.match(FILE_NAME_VALID_PATTERN, path):
        raise RuntimeError("The file path contains special characters.")


def check_path_readability(path):
    if not os.access(path, os.R_OK):
        raise RuntimeError("The file path is not readable.")


def check_path_writability(path):
    if not os.access(path, os.W_OK):
        raise RuntimeError("The file path is not writable.")


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


def check_file_size(file_path, max_size=FILE_MAX_SIZE):
    file_size = os.path.getsize(file_path)
    if file_size >= max_size:
        _user_interactive_confirm(f'The size of file path {file_path} exceeds {max_size} bytes.'
                                  f'Do you want to continue?')


def check_path_exists(path):
    if not os.path.exists(path):
        raise RuntimeError("The file path does not exist.")


def check_file_valid(path):
    check_path_exists(path)
    check_link(path)
    check_path_length(path)
    check_path_pattern_vaild(path)
    check_file_size(path)


def check_file_valid_readable(path):
    check_file_valid(path)
    check_path_readability(path)


def check_file_valid_writable(path):
    check_file_valid(path)
    check_path_writability(path)