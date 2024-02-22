import os
import time
import yaml
import torch
import pandas as pd
from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileOpen, create_directory, \
    check_link, FileChecker, FileCheckConst


class ListCache(list):
    threshold = 1000

    def __init__(self, *args):
        self._dump_count = 0
        super().__init__(*args)

    def __del__(self):
        self.flush()

    def flush(self):
        if len(self) == 0:
            return
        if not self._output_file:
            print("dumpfile path is not setted")
        write_csv(self._output_file, self, [])
        print(f"write {len(self)} items to {self._output_file} the {self._dump_count} time")
        self.clear()

    def append(self, data):
        list.append(self, data)
        if len(self) >= ListCache.threshold:
            self.flush()
    
    def set_output_file(self, output_file):
        self._output_file = output_file


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


def make_file_safety(file_path: str, permission=0o640):
    if os.path.islink(file_path):
        raise RuntimeError("Invalid soft link path: {}".format(file_path))
    file_real_path = os.path.realpath(file_path)
    if os.path.exists(file_real_path):
        return
    parent_path = os.path.dirname(file_real_path)
    if not os.path.exists(parent_path):
        create_directory(parent_path)
    if not os.access(parent_path, os.W_OK):
        raise PermissionError("The path {} is not writable!".format(parent_path))
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


def localtime_str():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def make_localtime_dir(path):
    if not os.path.isdir(path):
        create_directory(path)
    localtime_dir = os.path.join(path, localtime_str())
    create_directory(localtime_dir)
    return localtime_dir


def get_tensor_rank(x):
    if isinstance(x, (list, tuple)):
        if len(x) > 0:
            return get_tensor_rank(x[0])
        return None
    elif isinstance(x, torch.Tensor):
        device = x.device
        if device.type == 'cpu':
            return None
        else:
            return device.index
    return None


def get_rank_id(tensor):
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    rank = get_tensor_rank(tensor)
    if rank is not None:
        return rank
    return os.getpid()
