from ..common import print_warn_log_rank_0, seed_all
from ...core.utils import Const

class DebuggerConfig:
    def __init__(self, common_config, task_config, task, dump_path, level):
        self.dump_path = dump_path if dump_path else common_config.dump_path
        self.task = task or common_config.task or Const.STATISTICS
        self.rank = common_config.rank if common_config.rank else []
        self.step = common_config.step if common_config.step else []
        self.level = level or common_config.level or "L1"
        self.seed = common_config.seed if common_config.seed else 1234
        self.is_deterministic = common_config.is_deterministic if common_config.is_deterministic else False
        self.scope = task_config.scope if task_config.scope else []
        self.list = task_config.list if task_config.list else []
        self.data_mode =  task_config.data_mode if task_config.data_mode else ["all"]
        self.backward_input = task_config.backward_input
        self.summary_mode = task_config.summary_mode if task_config.summary_mode else Const.STATISTICS
        self.overflow_num = task_config.overflow_num if task_config.overflow_num else 1
        self.repair_scope = None
        self.repair_api_str = None
        self.on_step_end = None
        self.repair_type = None
        
        if self.task == "free_benchmark":
            self.fuzz_device = task_config.fuzz_device if task_config.fuzz_device else 'npu'
            self.handler_type = task_config.handler_type if task_config.handler_type else 'check'
            self.pert_mode = task_config.pert_mode if task_config.pert_mode else 'improve_precision'
            self.fuzz_level = task_config.fuzz_level if task_config.fuzz_level else 'L1'
            self.fuzz_stage = task_config.fuzz_stage if task_config.fuzz_stage else 'forward'
            self.preheat_config = {
                "if_preheat": task_config.if_preheat if task_config.if_preheat is not None else True, 
                "preheat_step": task_config.preheat_step if task_config.preheat_step else 15, 
                "max_sample": task_config.max_sample if task_config.max_sample else 20, 
            }
            
        self.check()
        if self.step:
            self.step.sort()
        seed_all(self.seed, self.is_deterministic)

    def check_kwargs(self):
        if self.task and self.task not in Const.TASK_LIST:
            raise Exception("task is invalid")
        if self.level and self.level not in Const.LEVEL_LIST:
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
                    raise ValueError(f"rank {self.rank} must be an integer and greater than or equal to 0.")
            else:
                print_warn_log_rank_0(f"Rank argument is provided. Only rank {self.rank} data will be dumpped.")

    def _check_step(self):
        if self.step:
            for s in self.step:
                if not isinstance(s, int) or s < 0:
                    raise ValueError(f"step element {s} must be an integer and greater than or equal to 0.")
    
    def check_model(self, model):
        if self.level in ["L0", "mix"] and not model:
            raise Exception(
                f"For level {self.level}, PrecisionDebugger must receive a model argument.",
            )