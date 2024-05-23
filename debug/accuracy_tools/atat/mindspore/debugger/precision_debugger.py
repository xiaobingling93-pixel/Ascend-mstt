import os
from atat.mindspore.ms_config import parse_json_config
from atat.mindspore.debugger.debugger_config import DebuggerConfig
from atat.mindspore.task_handler_factory import TaskHandlerFactory


class PrecisionDebugger:
    _instance = None

    def __new__(cls, config_path=None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
            cls._instance.config = None
        return cls._instance

    def __init__(self, config_path=None):
        if self.initialized:
            return
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), "../../config/config.json")
        common_config, task_config = parse_json_config(config_path)
        self.config = DebuggerConfig(common_config, task_config)
        self.initialized = True

    @classmethod
    def start(cls, target=None):
        instance = cls._instance
        if not instance:
            raise Exception("No instance of PrecisionDebugger found.")
        handler = TaskHandlerFactory.create(instance.config)
        handler.handle()
