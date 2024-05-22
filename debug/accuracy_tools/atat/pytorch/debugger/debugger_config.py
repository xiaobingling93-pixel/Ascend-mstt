from ..common import print_warn_log_rank_0, seed_all
from ...core.utils import Const


class DebuggerConfig:
    def __init__(self, common_config, task_config, task, dump_path, level):
        self.dump_path = dump_path if dump_path else common_config.dump_path
        self.task = task or common_config.task or "statistics"
        self.rank = common_config.rank if common_config.rank else []
        self.step = common_config.step if common_config.step else []
        self.level = level or common_config.level or "L1"
        self.seed = common_config.seed if common_config.seed else 1234
        self.is_deterministic = common_config.is_deterministic if common_config.is_deterministic else False
        self.scope = task_config.scope if task_config.scope else []
        self.list = task_config.list if task_config.list else []
        self.data_mode =  task_config.data_mode if task_config.data_mode else ["all"]
        self.backward_input = task_config.backward_input
        self.summary_mode = task_config.summary_mode if task_config.summary_mode else "statistics"
        self.overflow_num = task_config.overflow_num if task_config.overflow_num else 1
        self.repair_scope = None
        self.repair_api_str = None
        self.on_step_end = None

        self.check()
        if self.step:
            self.step.sort()
        seed_all(self.seed, self.is_deterministic)

    def check_kwargs(self):
        if self.task is not None and self.task not in Const.TASK_LIST:
            raise Exception("task is invalid")
        if self.level is not None and self.level not in ["L0", "L1", "L2"]:
            raise Exception("level is invalid")
        if not self.dump_path:
            raise Exception("Invalid dump path, please check your config")

    def check(self):
        self.check_kwargs()
        self._check_rank()
        self._check_step()
        return True

    def _check_rank(self):
        if self.rank:
            for rank_id in self.rank:
                if not isinstance(rank_id, int) or rank_id < 0:
                    raise ValueError(f"rank {self.rank} must be a positive integer.")
        else:
            print_warn_log_rank_0(f"Rank argument is provided. Only rank {self.rank} data will be dumpped.")

    def _check_step(self):
        if self.step:
            for s in self.step:
                if not isinstance(s, int):
                    raise ValueError(f"step element {s} should be int")