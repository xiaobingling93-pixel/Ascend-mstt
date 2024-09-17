from msprobe.pytorch.common import seed_all
from msprobe.core.common.const import Const


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
        self.overflow_nums = task_config.overflow_nums if task_config.overflow_nums else 1
        self.framework = Const.PT_FRAMEWORK

        if self.task == Const.FREE_BENCHMARK:
            self.fuzz_device = task_config.fuzz_device
            self.handler_type = task_config.handler_type
            self.pert_mode = task_config.pert_mode
            self.fuzz_level = task_config.fuzz_level
            self.fuzz_stage = task_config.fuzz_stage
            self.preheat_config = {
                "if_preheat": task_config.if_preheat, 
                "preheat_step": task_config.preheat_step, 
                "max_sample": task_config.max_sample
            }

        self.online_run_ut = False
        if self.task == Const.TENSOR:
            # dump api tensor and collaborate with online run_ut
            self.online_run_ut = task_config.online_run_ut if task_config.online_run_ut else False
            self.nfs_path = task_config.nfs_path if task_config.nfs_path else ""
            self.tls_path = task_config.tls_path if task_config.tls_path else ""
            self.host = task_config.host if task_config.host else ""
            self.port = task_config.port if task_config.port else -1

        self.check()

        if self.level == "L2":
            if not self.scope or not isinstance(self.scope, list) or len(self.scope) != 1:
                raise ValueError("scope must be configured as a list with one api name")
            if isinstance(self.scope[0], str) and Const.BACKWARD in self.scope[0] and not self.backward_input_list:
                raise ValueError("backward_input must be configured when scope contains 'backward'")
            if Const.BACKWARD in self.scope[0]:
                self.is_forward_acl_dump = False
                for index, scope_spec in enumerate(self.scope):
                    self.scope[index] = scope_spec.replace(Const.BACKWARD, Const.FORWARD)
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
        return True

    def check_model(self, model):
        if self.level in ["L0", "mix"] and not model:
            raise Exception(
                f"For level {self.level}, PrecisionDebugger must receive a model argument."
            )
