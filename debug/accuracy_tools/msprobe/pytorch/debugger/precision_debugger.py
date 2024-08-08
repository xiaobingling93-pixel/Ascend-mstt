import torch
from torch.utils.data import dataloader
from msprobe.pytorch.debugger.debugger_config import DebuggerConfig
from msprobe.pytorch.service import Service
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.pt_config import parse_json_config
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.const import Const
from msprobe.pytorch.grad_probe.grad_monitor import GradientMonitor


class PrecisionDebugger:
    _instance = None
    tasks_not_need_debugger = [Const.GRAD_PROBE]

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
            self.api_origin = False
            self.initialized = True
            self.model = self.check_model_valid(model)
            common_config, task_config = parse_json_config(config_path, task)
            self.task = common_config.task
            if self.task == Const.GRAD_PROBE:
                self.gm = GradientMonitor(common_config, task_config)
                return
            if step:
                common_config.step = step
            self.config = DebuggerConfig(
                common_config, task_config, task, dump_path, level
            )
            self.config.check_model(self.model)
            self.service = Service(self.config)
            self.enable_dataloader = self.config.enable_dataloader
            if self.enable_dataloader:
                logger.warning_on_rank_0("The enable_dataloader feature will be deprecated in the future.")
                dataloader._BaseDataLoaderIter.__next__ = iter_tracer(dataloader._BaseDataLoaderIter.__next__)

    @property
    def instance(self):
        return self._instance

    @staticmethod
    def check_model_valid(model):
        if not model or isinstance(model, torch.nn.Module):
            return model
        raise MsprobeException(
            MsprobeException.INVALID_PARAM_ERROR, "model 参数必须是torch.nn.Module类型。"
        )

    @classmethod
    def start(cls):
        instance = cls._instance
        if instance.task in PrecisionDebugger.tasks_not_need_debugger:
            return
        if not instance:
            raise Exception("No instance of PrecisionDebugger found.")
        if instance.enable_dataloader:
            logger.warning_on_rank_0("DataLoader is enabled, start() skipped.")
        else:
            instance.service.start(instance.model, instance.api_origin)
            instance.api_origin = False

    # 指定代码段dump前反向结束符，之后的计算过程数据将被忽略，无法被dump
    @classmethod
    def forward_backward_dump_end(cls):
        instance = cls._instance
        instance.service.forward_backward_dump_end()
        instance.api_origin = True

    @classmethod
    def stop(cls):
        instance = cls._instance
        if instance.task in PrecisionDebugger.tasks_not_need_debugger:
            return
        if not instance:
            raise Exception("PrecisionDebugger instance is not created.")
        if instance.enable_dataloader:
            logger.warning_on_rank_0("DataLoader is enabled, stop() skipped.")
        else:
            instance.service.stop()

    @classmethod
    def step(cls):
        if cls._instance.task in PrecisionDebugger.tasks_not_need_debugger:
            return
        if not cls._instance:
            raise Exception("PrecisionDebugger instance is not created.")
        cls._instance.service.step()

    @classmethod
    def monitor(cls, model):
        if not cls._instance:
            raise Exception("PrecisionDebugger instance is not created.")
        if cls._instance.task != Const.GRAD_PROBE:
            return
        cls._instance.gm.monitor(model)


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
