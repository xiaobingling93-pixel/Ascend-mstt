import os

import mindspore as ms

from msprobe.mindspore.service import Service
from msprobe.mindspore.ms_config import parse_json_config
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.task_handler_factory import TaskHandlerFactory
from msprobe.core.common.const import Const
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.runtime import Runtime

from msprobe.mindspore.grad_probe.grad_monitor import GradientMonitor


class PrecisionDebugger:
    _instance = None
    task_not_need_service = [Const.GRAD_PROBE]

    def __new__(cls, config_path=None, opt=None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
            cls._instance.config = None
            cls.service = None
            cls.first_start = False
        return cls._instance

    def __init__(self, config_path=None):
        if self.initialized:
            return
        self.initialized = True
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), "../../config/config.json")
        common_config, task_config = parse_json_config(config_path)
        self.task = common_config.task
        if self.task == Const.GRAD_PROBE:
            self.gm = GradientMonitor(common_config, task_config)
            return
        self.config = DebuggerConfig(common_config, task_config)

        Runtime.step_count = 0
        Runtime.is_running = False

    @staticmethod
    def _get_execution_mode():
        if ms.get_context("mode") == ms.GRAPH_MODE:
            if ms.context.get_jit_config().get("jit_level") == "O2" or ms.get_context("jit_level") == "O2":
                return MsConst.GRAPH_GE_MODE
            else:
                return MsConst.GRAPH_KBYK_MODE
        else:
            return MsConst.PYNATIVE_MODE

    @classmethod
    def start(cls, target=None):
        instance = cls._instance
        if not instance:
            raise Exception("No instance of PrecisionDebugger found.")
        if instance.task in PrecisionDebugger.task_not_need_service:
            return

        instance.config.execution_mode = instance._get_execution_mode()
        if instance.config.execution_mode == MsConst.PYNATIVE_MODE and instance.config.task != Const.FREE_BENCHMARK:
            if not instance.service:
                instance.service = Service(instance.config)
            instance.service.start(target)
        else:
            if not instance.first_start:
                handler = TaskHandlerFactory.create(instance.config)
                handler.handle()

        instance.first_start = True
        Runtime.is_running = True

    @classmethod
    def stop(cls):
        instance = cls._instance
        if not instance:
            raise Exception("PrecisionDebugger instance is not created.")
        if instance.task == Const.GRAD_PROBE:
            instance.gm.stop()
        if instance.task in PrecisionDebugger.task_not_need_service:
            return
        if instance.service:
            instance.service.stop()
        Runtime.is_running = False

    @classmethod
    def step(cls):
        instance = cls._instance
        if not instance:
            raise Exception("PrecisionDebugger instance is not created.")
        if instance.task in PrecisionDebugger.task_not_need_service:
            return
        if instance.service:
            instance.service.step()
        Runtime.step_count += 1

    @classmethod
    def monitor(cls, opt):
        instance = cls._instance
        if not instance:
            raise Exception("PrecisionDebugger instance is not created.")
        if instance.task != Const.GRAD_PROBE:
            return
        instance.gm.monitor(opt)
