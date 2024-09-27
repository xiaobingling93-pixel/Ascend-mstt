from msprobe.core.common.const import Const, FileCheckConst
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.file_utils import FileChecker
from msprobe.core.common.utils import get_real_step_or_rank


class CommonConfig:
    def __init__(self, json_config):
        self.task = json_config.get('task')
        self.dump_path = json_config.get('dump_path')
        self.rank = get_real_step_or_rank(json_config.get('rank'), Const.RANK)
        self.step = get_real_step_or_rank(json_config.get('step'), Const.STEP)
        self.level = json_config.get('level')
        self.acl_config = json_config.get('acl_config')
        self.enable_dataloader = json_config.get('enable_dataloader', False)
        self._check_config()

    def _check_config(self):
        if self.task and self.task not in Const.TASK_LIST:
            logger.error_log_with_exp("task is invalid, it should be one of {}".format(Const.TASK_LIST),
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if self.dump_path is not None and not isinstance(self.dump_path, str):
            logger.error_log_with_exp("dump_path is invalid, it should be a string",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if self.level and self.level not in Const.LEVEL_LIST:
            logger.error_log_with_exp("level is invalid, it should be one of {}".format(Const.LEVEL_LIST),
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if not isinstance(self.enable_dataloader, bool):
            logger.error_log_with_exp("enable_dataloader is invalid, it should be a boolean",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        if self.acl_config:
            self._check_acl_config()

    def _check_acl_config(self):
        if not isinstance(self.acl_config, str):
            logger.error_log_with_exp("acl_config is invalid, it should be a string",
                                      MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
        file_checker = FileChecker(
            file_path=self.acl_config, path_type=FileCheckConst.FILE, file_type=FileCheckConst.JSON_SUFFIX)
        file_checker.common_check()


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

    @staticmethod
    def _check_str_list_config(config_item, config_name):
        if config_item is not None:
            if not isinstance(config_item, list):
                logger.error_log_with_exp(f"{config_name} is invalid, it should be a list[str]",
                                          MsprobeException(MsprobeException.INVALID_PARAM_ERROR))
            for name in config_item:
                if not isinstance(name, str):
                    logger.error_log_with_exp(f"{config_name} is invalid, it should be a list[str]",
                                              MsprobeException(MsprobeException.INVALID_PARAM_ERROR))

    def check_config(self):
        self._check_str_list_config(self.scope, "scope")
        self._check_str_list_config(self.list, "list")
        self._check_str_list_config(self.data_mode, "data_mode")
        self._check_str_list_config(self.backward_input, "backward_input")
