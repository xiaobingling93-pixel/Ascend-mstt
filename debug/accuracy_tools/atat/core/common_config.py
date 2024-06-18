from .utils import Const


# 公共配置类
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
            raise Exception("task is invalid")
        if self.rank is not None and not isinstance(self.rank, list):
            raise Exception("rank is invalid")
        if self.step is not None and not isinstance(self.step, list):
            raise Exception("step is invalid")
        if self.level and self.level not in Const.LEVEL_LIST:
            raise Exception("level is invalid")
        if self.seed is not None and not isinstance(self.seed, int):
            raise Exception("seed is invalid")
        if not isinstance(self.is_deterministic, bool):
            raise Exception("is_deterministic is invalid")
        if not isinstance(self.enable_dataloader, bool):
            raise Exception("enable_dataloader is invalid")
        

# 基础配置类
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
            raise Exception("scope is invalid")
        if self.list is not None and not isinstance(self.list, list):
            raise Exception("list is invalid")
        if self.data_mode is not None and not isinstance(self.data_mode, list):
            raise Exception("data_mode is invalid")
        