import os

import mindspore as ms

from msprobe.mindspore.service import Service
from msprobe.mindspore.ms_config import parse_json_config
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.task_handler_factory import TaskHandlerFactory
from msprobe.core.common.const import Const, MsConst
from msprobe.mindspore.runtime import Runtime


class PrecisionDebugger:
    _instance = None

    def __new__(cls, config_path=None):
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
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), "../../config/config.json")
        common_config, task_config = parse_json_config(config_path)
        self.config = DebuggerConfig(common_config, task_config)
        self.initialized = True
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
    def start(cls):
        instance = cls._instance
        if not instance:
            raise Exception("No instance of PrecisionDebugger found.")

        instance.config.execution_mode = instance._get_execution_mode()
        if instance.config.execution_mode == MsConst.PYNATIVE_MODE and instance.config.level == MsConst.API and \
           instance.config.task != Const.FREE_BENCHMARK:
            if not instance.service:
                instance.service = Service(instance.config)
            instance.service.start()
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
        if instance.service:
            instance.service.stop()
        Runtime.is_running = False

    @classmethod
    def step(cls):
        instance = cls._instance
        if not instance:
            raise Exception("PrecisionDebugger instance is not created.")
        if instance.service:
            instance.service.step()
        Runtime.step_count += 1
