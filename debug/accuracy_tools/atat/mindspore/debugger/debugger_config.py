import os
from atat.core.common.utils import Const


class DebuggerConfig:
    convert_map = {
        "L0": "cell",
        "L1": "api",
        "L2": 'kernel'
    }

    def __init__(self, common_config, task_config):
        self.dump_path = common_config.dump_path
        self.task = common_config.task
        self.rank = [] if not common_config.rank else common_config.rank
        self.step = [] if not common_config.step else common_config.step
        if not common_config.level:
            common_config.level = "L1"
        self.level = DebuggerConfig.convert_map[common_config.level]
        self.level_ori = common_config.level
        self.list = [] if not task_config.list else task_config.list
        self.scope =[] if not task_config.scope else task_config.scope
        self.data_mode =  [] if not task_config.data_mode else task_config.data_mode
        self.file_format = task_config.file_format
        self.check_mode = task_config.check_mode
        self.framework = Const.MS_FRAMEWORK
        self.summary_mode = task_config.summary_mode
        self.check()

    def check(self):
        if not self.dump_path:
            raise Exception("Dump path is empty.")
        if self.level_ori != "L1" and not os.path.isabs(self.dump_path):
            raise Exception("Dump path must be absolute path.")
        if not self.task:
            self.task = "statistics"
        if not self.level:
            raise Exception("level must be L0, L1 or L2")
        if not self.file_format:
            self.file_format = "npy"
        if not self.check_mode:
            self.check_mode = "all"
        self._check_rank()
        self._check_step()
        return True

    def _check_rank(self):
        for rank_id in self.rank:
            if not isinstance(rank_id, int) or rank_id < 0:
                raise ValueError(f"rank {self.rank} must be a positive integer.")

    def _check_step(self):
        for s in self.step:
            if not isinstance(s, int):
                raise ValueError(f"step element {s} should be int")
