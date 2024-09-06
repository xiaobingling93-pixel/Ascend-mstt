from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException


class CommonConfig:
    def __init__(self, json_config):
        self.task = json_config.get('task')
        self.dump_path = json_config.get('dump_path')
        self.rank = json_config.get('rank')
        self.step = self.get_real_step(json_config.get('step'))
        self.level = json_config.get('level')
        self.seed = json_config.get('seed')
        self.acl_config = json_config.get('acl_config')
        self.is_deterministic = json_config.get('is_deterministic', False)
        self.enable_dataloader = json_config.get('enable_dataloader', False)
        self._check_config()

    @staticmethod
    def get_step_from_string(step):
        try:
            borderline = int(step.split('-')[0]), int(step.split('-')[-1])
        except ValueError as e:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR, 
                                   "The connector(-) must start and end with decimal numbers.") from e
        if borderline[0] <= borderline[1]:
            continual_step = list(range(borderline[0], borderline[1] + 1))
        else:
            continual_step = list(range(borderline[0], borderline[1] - 1, -1))
        return continual_step

    def get_real_step(self, step_input):
        if step_input is not None and not isinstance(step_input, list):
            logger.error_log_with_exp("step is invalid, it should be a list",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        real_step = []
        for step in step_input:
            if not isinstance(step, (int, str)):
                    raise ValueError(f"step element {step} must be an integer or string.")
            if isinstance(step, int) and step >= 0:
                real_step.append(step)
            elif isinstance(step, str) and '-' in step:
                continual_step = self.get_step_from_string(step)
                real_step.extend(continual_step)
        real_step.sort()
        return real_step
    
    def _check_config(self):
        if self.task and self.task not in Const.TASK_LIST:
            logger.error_log_with_exp("task is invalid, it should be one of {}".format(Const.TASK_LIST),
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if self.rank is not None and not isinstance(self.rank, list):
            logger.error_log_with_exp("rank is invalid, it should be a list",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if self.level and self.level not in Const.LEVEL_LIST:
            logger.error_log_with_exp("level is invalid, it should be one of {}".format(Const.LEVEL_LIST),
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if self.seed is not None and not isinstance(self.seed, int):
            logger.error_log_with_exp("seed is invalid, it should be an integer",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if not isinstance(self.is_deterministic, bool):
            logger.error_log_with_exp("is_deterministic is invalid, it should be a boolean",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if not isinstance(self.enable_dataloader, bool):
            logger.error_log_with_exp("enable_dataloader is invalid, it should be a boolean",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))


class BaseConfig:
    def __init__(self, json_config):
        self.scope = json_config.get('scope')
        self.list = json_config.get('list')
        self.data_mode = json_config.get('data_mode')
        self.backward_input = json_config.get("backward_input")
        self.file_format = json_config.get("file_format")
        self.summary_mode = json_config.get("summary_mode")
        self.overflow_nums = json_config.get("overflow_nums")
        self.check_mode = json_config.get("check_mode")
        self.fuzz_device = json_config.get("fuzz_device")
        self.pert_mode = json_config.get("pert_mode")
        self.handler_type = json_config.get("handler_type")
        self.fuzz_level = json_config.get("fuzz_level")
        self.fuzz_stage = json_config.get("fuzz_stage")
        self.if_preheat = json_config.get("if_preheat")
        self.preheat_step = json_config.get("preheat_step")
        self.max_sample = json_config.get("max_sample")

    def check_config(self):
        if self.scope is not None and not isinstance(self.scope, list):
            logger.error_log_with_exp("scope is invalid, it should be a list",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if self.list is not None and not isinstance(self.list, list):
            logger.error_log_with_exp("list is invalid, it should be a list",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if self.data_mode is not None and not isinstance(self.data_mode, list):
            logger.error_log_with_exp("data_mode is invalid, it should be a list",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
