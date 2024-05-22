from .debugger_config import DebuggerConfig
from ..service import Service
from ..common import print_warn_log_rank_0


class PrecisionDebugger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PrecisionDebugger, cls).__new__(cls)
            cls._instance.config = None
            cls._instance.model = None
            cls._instance.enable_dataloader = False
        return cls._instance

    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.config = DebuggerConfig(*args, **kwargs)

            self.service = Service(self.config)  # todo: enable dataloader功能

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
