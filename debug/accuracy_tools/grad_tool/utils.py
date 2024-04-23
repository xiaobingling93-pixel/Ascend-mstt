import os
import yaml
import torch
import torch.distributed as dist
import pandas as pd
from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileOpen, create_directory, \
    FileChecker, FileCheckConst
from ptdbg_ascend.src.python.ptdbg_ascend.common.utils import check_file_or_directory_path, print_info_log, \
    print_warn_log


def get_config(filepath):
    with FileOpen(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config


def write_csv(filepath, content_list, header):
    if not os.path.exists(filepath):
        make_file_safety(filepath)
        data_frame = pd.DataFrame(columns=header)
        data_frame.to_csv(filepath, index=False)

    filepath_checker = FileChecker(filepath, FileCheckConst.FILE)
    filepath_checker.common_check()
    new_data = pd.DataFrame(list(content for content in content_list))
    new_data.to_csv(filepath, mode='a+', header=False, index=False)
    print_info_log(f"write {len(content_list)} items to {filepath}")


def make_file_safety(file_path: str, permission=0o640):
    if os.path.islink(file_path):
        raise RuntimeError(f"Invalid soft link path: {file_path}")
    file_real_path = os.path.realpath(file_path)
    if os.path.exists(file_real_path):
        return
    parent_path = os.path.dirname(file_real_path)
    if not os.path.exists(parent_path):
        create_directory(parent_path)
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


def get_rank_id():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return os.getpid()


def path_check(path, isdir=False):
    check_file_or_directory_path(path, isdir)


def print_rank_0(message):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message)
    else:
        print(message)
