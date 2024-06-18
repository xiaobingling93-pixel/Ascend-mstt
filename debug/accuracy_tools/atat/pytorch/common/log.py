import os
import time
import sys
from .utils import get_rank_if_initialized


def on_rank_0(func):
    def func_rank_0(*args, **kwargs):
        current_rank = get_rank_if_initialized()
        if current_rank is None or current_rank == 0:
            return func(*args, **kwargs)

    return func_rank_0


def _print_log(level, msg, end='\n'):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getpid()
    full_msg = current_time + "(" + str(pid) + ")-[" + level + "]" + msg
    current_rank = get_rank_if_initialized()
    if current_rank is not None:
        full_msg = f"[rank {current_rank}]-" + full_msg
    print(full_msg, end=end)
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


print_info_log_rank_0 = on_rank_0(print_info_log)
print_warn_log_rank_0 = on_rank_0(print_warn_log)
print_error_log_rank_0 = on_rank_0(print_error_log)
