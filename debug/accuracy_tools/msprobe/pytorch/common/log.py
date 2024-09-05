import os
import time
import sys
from msprobe.pytorch.common.utils import get_rank_if_initialized
from msprobe.core.common.log import BaseLogger
from msprobe.core.common.exceptions import DistributedNotInitializedError


class PyTorchLogger(BaseLogger):
    def __init__(self):
        super().__init__()

    def get_rank(self):
        try:
            current_rank = get_rank_if_initialized()
        except DistributedNotInitializedError:
            current_rank = None
        return current_rank


logger = PyTorchLogger()