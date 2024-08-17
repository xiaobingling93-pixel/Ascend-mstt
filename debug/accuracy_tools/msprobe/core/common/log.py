import os
import time
import sys
from functools import wraps
from msprobe.core.common.const import MsgConst


class BaseLogger:
    def __init__(self):
        self.warning_level = "WARNING"
        self.error_level = "ERROR"
        self.info_level = "INFO"
        self.rank = None

    @staticmethod
    def _print_log(level, msg, end='\n'):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        pid = os.getpid()
        full_msg = f"{current_time} ({pid}) [{level}] {msg}"
        print(full_msg, end=end)
        sys.stdout.flush()

    def get_rank(self):
        return self.rank

    def filter_special_chars(func):
        @wraps(func)
        def wrapper(self, msg):
            for char in MsgConst.SPECIAL_CHAR:
                msg = msg.replace(char, '_')
            return func(self, msg)
        return wrapper

    @filter_special_chars
    def info(self, msg):
        self._print_log(self.info_level, msg)

    @filter_special_chars
    def error(self, msg):
        self._print_log(self.error_level, msg)

    @filter_special_chars
    def warning(self, msg):
        self._print_log(self.warning_level, msg)

    def on_rank_0(self, func):
        def func_rank_0(*args, **kwargs):
            current_rank = self.get_rank()
            if current_rank is None or current_rank == 0:
                return func(*args, **kwargs)
            else:
                return None
        return func_rank_0

    def info_on_rank_0(self, msg):
        return self.on_rank_0(self.info)(msg)

    def error_on_rank_0(self, msg):
        return self.on_rank_0(self.error)(msg)

    def warning_on_rank_0(self, msg):
        return self.on_rank_0(self.warning)(msg)
    
    def error_log_with_exp(self, msg, exception):
        self.error(msg)
        raise exception


logger = BaseLogger()
