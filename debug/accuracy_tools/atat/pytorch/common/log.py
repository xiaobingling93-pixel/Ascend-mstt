import os
import time
import sys
from atat.pytorch.common.utils import get_rank_if_initialized
from atat.core.common.log import BaseLogger
from atat.core.common.exceptions import DistributedNotInitializedError


class PyTorchLogger(BaseLogger):
    def __init__(self):
        super().__init__()

    def get_rank(self):
        try:
            current_rank = get_rank_if_initialized()
        except DistributedNotInitializedError:
            current_rank = None
        return current_rank

    def _print_log(self, level, msg, end='\n'):
        current_rank = self.get_rank()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        pid = os.getpid()
        if current_rank is not None:
            full_msg = f"{current_time} ({pid}) [rank {current_rank}] [{level}] {msg}"
        else:
            full_msg = f"{current_time} ({pid}) [{level}] {msg}"
        print(full_msg, end=end)
        sys.stdout.flush()


logger = PyTorchLogger()