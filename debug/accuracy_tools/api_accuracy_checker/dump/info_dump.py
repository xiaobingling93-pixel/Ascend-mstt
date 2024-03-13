import fcntl
import json
import os
import threading
import multiprocessing

from api_accuracy_checker.dump.api_info import ForwardAPIInfo, BackwardAPIInfo
from api_accuracy_checker.common.utils import check_file_or_directory_path, initialize_save_path, create_directory
from ptdbg_ascend.src.python.ptdbg_ascend.common.utils import check_path_before_create
from api_accuracy_checker.common.config import msCheckerConfig


from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileOpen, FileCheckConst, FileChecker, change_mode

lock = threading.Lock()
proc_lock = multiprocessing.Lock()


def write_api_info_json(api_info):
    from api_accuracy_checker.dump.dump import DumpUtil
    dump_path = msCheckerConfig.dump_path
    dump_path = os.path.join(msCheckerConfig.dump_path, "step" + str((DumpUtil.call_num - 1) if msCheckerConfig.enable_dataloader else DumpUtil.call_num)) 
    check_path_before_create(dump_path)
    create_directory(dump_path)
    rank = api_info.rank
    if isinstance(api_info, ForwardAPIInfo):
        file_path = os.path.join(dump_path, f'forward_info_{rank}.json')
        stack_file_path = os.path.join(dump_path, f'stack_info_{rank}.json')
        write_json(file_path, api_info.api_info_struct)
        write_json(stack_file_path, api_info.stack_info_struct, indent=4)

    elif isinstance(api_info, BackwardAPIInfo):
        file_path = os.path.join(dump_path, f'backward_info_{rank}.json')
        write_json(file_path, api_info.grad_info_struct)
    else:
        raise ValueError(f"Invalid api_info type {type(api_info)}")


def write_json(file_path, data, indent=None):
    check_file_or_directory_path(os.path.dirname(file_path), True)
    with lock, FileOpen(file_path, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            current_position = f.tell()
            if current_position > 0:
                f.seek(current_position - 1, os.SEEK_SET)
                f.truncate()
                if f.tell() > 3:
                    f.seek(f.tell() - 1, os.SEEK_SET)
                    f.truncate()
                    f.write(',\n')
                f.write(json.dumps(data, indent=indent)[1:-1] + '\n}')
            else:
                change_mode(file_path, FileCheckConst.DATA_FILE_AUTHORITY)
                f.write('{\n' + json.dumps(data, indent=indent)[1:] + '\n')
        except Exception as e:
            raise ValueError(f"Json save failed:{e}") from e
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def initialize_output_json():
    dump_path = msCheckerConfig.dump_path
    check_path_before_create(dump_path)
    create_directory(dump_path)
    dump_path_checker = FileChecker(dump_path, FileCheckConst.DIR)
    dump_path = dump_path_checker.common_check()
    files = ['forward_info.json', 'backward_info.json', 'stack_info.json']
    for file in files:
        file_path = os.path.join(dump_path, file)
        if os.path.exists(file_path):
            raise ValueError(f"file {file_path} already exists, please remove it first or use a new dump path")
