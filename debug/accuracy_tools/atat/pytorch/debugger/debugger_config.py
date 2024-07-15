from atat.pytorch.common import seed_all
from atat.pytorch.common.log import logger
from atat.core.common.utils import Const


class DebuggerConfig:
    def __init__(self, common_config, task_config, task, dump_path, level):
        self.dump_path = dump_path if dump_path else common_config.dump_path
        self.task = task or common_config.task or Const.STATISTICS
        self.rank = common_config.rank if common_config.rank else []
        self.step = common_config.step if common_config.step else []
        self.level = level or common_config.level or "L1"
        self.seed = common_config.seed if common_config.seed else 1234
        self.is_deterministic = common_config.is_deterministic
        self.enable_dataloader = common_config.enable_dataloader
        self.scope = task_config.scope if task_config.scope else []
        self.list = task_config.list if task_config.list else []
        self.data_mode = task_config.data_mode if task_config.data_mode else ["all"]
        self.backward_input_list = task_config.backward_input if task_config.backward_input else []
        self.backward_input = {}
        self.acl_config = common_config.acl_config if common_config.acl_config else ""
        self.is_forward_acl_dump = True
        self.summary_mode = task_config.summary_mode if task_config.summary_mode else Const.STATISTICS
        self.overflow_num = task_config.overflow_num if task_config.overflow_num else 1
        self.framework = Const.PT_FRAMEWORK

        if self.task == Const.FREE_BENCHMARK:
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
        if self.level == "L2":
            if not self.scope or not isinstance(self.scope, list) or len(self.scope) != 1:
                raise ValueError("scope must be configured as a list with one api name")
            if isinstance(self.scope[0], str) and Const.BACKWARD in self.scope[0] and not self.backward_input_list:
                raise ValueError("backward_input must be configured when scope contains 'backward'")
            if Const.BACKWARD in self.scope[0]:
                self.is_forward_acl_dump = False
                for index in range(len(self.scope)):
                    # Do this replace operation to let the acl backward dump can be done in forward hook.
                    self.scope[index] = self.scope[index].replace(Const.BACKWARD, Const.FORWARD)
                    self.backward_input[self.scope[index]] = self.backward_input_list[index]
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

    def check_model(self, model):
        if self.level in ["L0", "mix"] and not model:
            raise Exception(
                f"For level {self.level}, PrecisionDebugger must receive a model argument."
            )

    def _check_rank(self):
        if self.rank:
            for rank_id in self.rank:
                if not isinstance(rank_id, int) or rank_id < 0:
                    raise ValueError(f"rank {self.rank} must be an integer and greater than or equal to 0.")
            else:
                logger.warning_on_rank_0(f"Rank argument is provided. Only rank {self.rank} data will be dumpped.")

    def _check_step(self):
        if self.step:
            for s in self.step:
                if not isinstance(s, int) or s < 0:
                    raise ValueError(f"step element {s} must be an integer and greater than or equal to 0.")
