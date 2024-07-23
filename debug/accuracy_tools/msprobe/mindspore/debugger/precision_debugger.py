import os
import mindspore as ms
from msprobe.mindspore.service import Service
from msprobe.mindspore.ms_config import parse_json_config
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.task_handler_factory import TaskHandlerFactory


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
        self.service = Service(self.config)

    @classmethod
    def start(cls):
        instance = cls._instance
        if not instance:
            raise Exception("No instance of PrecisionDebugger found.")
        if ms.get_context("mode") == 1 and instance.config.level_ori == "L1":
            instance.service.start()
        else:
            handler = TaskHandlerFactory.create(instance.config)
            handler.handle()

    @classmethod
    def stop(cls):
        instance = cls._instance
        if not instance:
            raise Exception("PrecisionDebugger instance is not created.")
        instance.service.stop()

    @classmethod
    def step(cls):
        if not cls._instance:
            raise Exception("PrecisionDebugger instance is not created.")
        cls._instance.service.step()