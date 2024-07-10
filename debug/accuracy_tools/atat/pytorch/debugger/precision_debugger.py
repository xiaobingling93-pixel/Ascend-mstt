import torch
from torch.utils.data import dataloader
from atat.pytorch.debugger.debugger_config import DebuggerConfig
from atat.pytorch.service import Service
from atat.pytorch.common import print_warn_log_rank_0
from atat.pytorch.pt_config import parse_json_config
from atat.pytorch.common.exceptions import MsaccException


class PrecisionDebugger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PrecisionDebugger, cls).__new__(cls)
            cls._instance.config = None
            cls._instance.enable_dataloader = False
        return cls._instance

    def __init__(
        self,
        config_path=None,
        task=None,
        dump_path=None,
        level=None,
        model=None,
        step=None,
    ):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.model = self.check_model_valid(model)
            common_config, task_config = parse_json_config(config_path, task)
            if step:
                common_config.step = step
            self.config = DebuggerConfig(
                common_config, task_config, task, dump_path, level
            )
            self.config.check_model(self.model)
            self.service = Service(self.config)
            self.enable_dataloader = self.config.enable_dataloader
            if self.enable_dataloader:
                print_warn_log_rank_0("The enable_dataloader feature will be deprecated in the future.")
                dataloader._BaseDataLoaderIter.__next__ = iter_tracer(dataloader._BaseDataLoaderIter.__next__)

    @staticmethod
    def check_model_valid(model):
        if not model or isinstance(model, torch.nn.Module):
            return model
        raise MsaccException(
            MsaccException.INVALID_PARAM_ERROR, "model 参数必须是torch.nn.Module类型。"
        )

    @classmethod
    def start(cls):
        instance = cls._instance
        if not instance:
            raise Exception("No instance of PrecisionDebugger found.")
        if instance.enable_dataloader:
            print_warn_log_rank_0("DataLoader is enabled, start() skipped.")
        else:
            instance.service.start(instance.model)

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

    @property
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def iter_tracer(func):
    def func_wrapper(*args, **kwargs):
        debugger_instance = PrecisionDebugger.instance
        debugger_instance.enable_dataloader = False
        if not debugger_instance.service.first_start:
            debugger_instance.stop()
            debugger_instance.step()
        result = func(*args, **kwargs)
        debugger_instance.start()
        debugger_instance.enable_dataloader = True
        return result
    return func_wrapper
