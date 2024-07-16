from atat.core.common.utils import Const
from atat.core.common.log import logger
from atat.core.common.exceptions import MsaccException


class CommonConfig:
    def __init__(self, json_config):
        self.task = json_config.get('task')
        self.dump_path = json_config.get('dump_path')
        self.rank = json_config.get('rank')
        self.step = json_config.get('step')
        self.level = json_config.get('level')
        self.seed = json_config.get('seed')
        self.acl_config = json_config.get('acl_config')
        self.is_deterministic = json_config.get('is_deterministic', False)
        self.enable_dataloader = json_config.get('enable_dataloader', False)
        self._check_config()

    def _check_config(self):
        if self.task and self.task not in Const.TASK_LIST:
            logger.error_log_with_exp(
                "task is invalid, it should be one of {}".format(Const.TASK_LIST), MsaccException(MsaccException.INVALID_PARAM_ERROR))
        if self.rank is not None and not isinstance(self.rank, list):
            logger.error_log_with_exp("rank is invalid, it should be a list", MsaccException(MsaccException.INVALID_PARAM_ERROR))
        if self.step is not None and not isinstance(self.step, list):
            logger.error_log_with_exp("step is invalid, it should be a list", MsaccException(MsaccException.INVALID_PARAM_ERROR))
        if self.level and self.level not in Const.LEVEL_LIST:
            logger.error_log_with_exp(
                "level is invalid, it should be one of {}".format(Const.LEVEL_LIST), MsaccException(MsaccException.INVALID_PARAM_ERROR))
        if self.seed is not None and not isinstance(self.seed, int):
            logger.error_log_with_exp("seed is invalid, it should be an integer", MsaccException(MsaccException.INVALID_PARAM_ERROR))
        if not isinstance(self.is_deterministic, bool):
            logger.error_log_with_exp(
                "is_deterministic is invalid, it should be a boolean", MsaccException(MsaccException.INVALID_PARAM_ERROR))
        if not isinstance(self.enable_dataloader, bool):
            logger.error_log_with_exp(
                "enable_dataloader is invalid, it should be a boolean", MsaccException(MsaccException.INVALID_PARAM_ERROR))
        

class BaseConfig:
    def __init__(self, json_config):
        self.scope = json_config.get('scope')
        self.list = json_config.get('list')
        self.data_mode = json_config.get('data_mode')
        self.backward_input = json_config.get("backward_input")
        self.file_format = json_config.get("file_format")
        self.summary_mode =  json_config.get("summary_mode")
        self.overflow_num = json_config.get("overflow_num")
        self.check_mode = json_config.get("check_mode")

    def check_config(self):
        if self.scope is not None and not isinstance(self.scope, list):
            logger.error_log_with_exp("scope is invalid, it should be a list", MsaccException(MsaccException.INVALID_PARAM_ERROR))
        if self.list is not None and not isinstance(self.list, list):
            logger.error_log_with_exp("list is invalid, it should be a list", MsaccException(MsaccException.INVALID_PARAM_ERROR))
        if self.data_mode is not None and not isinstance(self.data_mode, list):
            logger.error_log_with_exp("data_mode is invalid, it should be a list", MsaccException(MsaccException.INVALID_PARAM_ERROR))
        
