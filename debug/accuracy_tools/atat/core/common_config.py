from .file_check_util import FileChecker, FileCheckConst
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
        self.is_deterministic = json_config.get('is_deterministic')
        self._check_config()

    def _check_config(self):
        if self.task is not None and self.task not in Const.TASK_LIST:
            raise Exception("task is invalid")
        if self.rank is not None and not isinstance(self.rank, list):
            raise Exception("rank is invalid")
        if self.step is not None and not isinstance(self.step, list):
            raise Exception("step is invalid")
        if self.level is not None and self.level not in ["L0", "L1", "L2"]:
            raise Exception("level is invalid")
        if self.seed is not None and not isinstance(self.seed, int):
            raise Exception("seed is invalid")
        if self.is_deterministic is not None and not isinstance(self.is_deterministic, bool):
            raise Exception("is_deterministic is invalid")
        

# 基础配置类
class BaseConfig:
    def __init__(self, json_config):
        self.scope = json_config.get('scope')
        self.list = json_config.get('list')
        self.data_mode = json_config.get('data_mode')

    def check_config(self):
        if self.scope is not None and not isinstance(self.scope, list):
            raise Exception("scope is invalid")
        if self.list is not None and not isinstance(self.list, list):
            raise Exception("list is invalid")
        if self.data_mode is not None and not isinstance(self.data_mode, list):
            raise Exception("data_mode is invalid")
        