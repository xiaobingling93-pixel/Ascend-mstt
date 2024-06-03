from .debugger_config import DebuggerConfig
from ..service import Service
from ..common import print_warn_log_rank_0
from ..pt_config import parse_json_config


class PrecisionDebugger:
    _instance = None

    def __new__(cls, config_path=None, task=None, dump_path=None, level=None):
        if cls._instance is None:
            cls._instance = super(PrecisionDebugger, cls).__new__(cls)
            cls._instance.config = None
            cls._instance.model = None
            cls._instance.enable_dataloader = False
        return cls._instance

    def __init__(self, config_path=None, task=None, dump_path=None, level=None):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            common_config, task_config = parse_json_config(config_path, task)
            self.config = DebuggerConfig(common_config, task_config, task, dump_path, level)
            self.service = Service(self.config)
    
    @classmethod
    def start(cls, model):
        instance = cls._instance
        if not instance:
            raise Exception("No instance of PrecisionDebugger found.")
        if instance.enable_dataloader:
            print_warn_log_rank_0("DataLoader is enabled, start() skipped.")
        else:
            instance.service.start(model)

    @classmethod
    def stop(cls):
        instance = cls._instance
        if not instance:
            raise Exception("PrecisionDebugger instance is not created.")
        if instance.enable_dataloader:
            print_warn_log_rank_0("DataLoader is enabled, stop() skipped.")
        else:
            instance.service.stop()

    @classmethod
    def step(cls):
        if not cls._instance:
            raise Exception("PrecisionDebugger instance is not created.")
        cls._instance.service.step()
