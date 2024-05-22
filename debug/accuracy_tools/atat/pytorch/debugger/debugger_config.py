import os
from ..common import print_warn_log_rank_0



class DebuggerConfig:
    def __init__(self, dump_path, task, level=None, scope=[], api_list=[], on_step_end=None,
                 rank=None, step=None, repair_type=None, repair_scope=None, repair_api_str=None,
                 task_config=None):
        self.task_config = task_config
        self.dump_path = dump_path
        self.task = task
        self.rank = rank
        self.step = step if step is not None else []
        self.scope = scope
        self.level = level
        self.api_list = api_list
        self.repair_type = repair_type
        self.repair_scope = repair_scope
        self.repair_api_str = repair_api_str
        self.on_step_end = on_step_end

        self.check()
        if self.step:
            self.step.sort()

    def check(self):
        # self._check_hook_name()
        self._check_rank()
        self._check_step()
        return True

    def _check_hook_name(self):
        if self.hook_name not in ["dump", "overflow_check"]:
            raise ValueError(f"hook_name should be in ['dump', 'overflow_check'], got {self.hook_name}")

    def _check_rank(self):
        if self.rank is not None:
            if not isinstance(self.rank, int) or self.rank < 0:
                raise ValueError(f"rank {self.rank} must be a positive integer.")
            else:
                print_warn_log_rank_0(f"Rank argument is provided. Only rank {self.rank} data will be dumpped.")

    def _check_step(self):
        if not isinstance(self.step, list):
            raise ValueError(f"step {self.step} should be list")
        for s in self.step:
            if not isinstance(s, int):
                raise ValueError(f"step element {s} should be int")
