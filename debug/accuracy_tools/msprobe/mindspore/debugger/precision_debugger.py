# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import mindspore as ms
from mindspore._c_expression import MSContext

from msprobe.core.common.const import Const, MsgConst
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.grad_probe.grad_monitor import GradientMonitor
from msprobe.mindspore.ms_config import parse_json_config
from msprobe.mindspore.runtime import Runtime
from msprobe.mindspore.service import Service
from msprobe.mindspore.task_handler_factory import TaskHandlerFactory


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
            config_path = os.path.join(os.path.dirname(__file__), "../../config.json")
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
        jit_level = ms.context.get_jit_config().get(MsConst.JIT_LEVEL)
        jit_level = jit_level if jit_level else ms.get_context(MsConst.JIT_LEVEL)
        if not jit_level:
            if MSContext.get_instance().get_ascend_soc_version() == MsConst.ASCEND_910A:
                jit_level = MsConst.JIT_LEVEL_O2
            else:
                jit_level = MsConst.JIT_LEVEL_O0

        if ms.get_context("mode") == ms.GRAPH_MODE:
            if jit_level == MsConst.JIT_LEVEL_O2:
                return MsConst.GRAPH_GE_MODE
            else:
                return MsConst.GRAPH_KBYK_MODE
        else:
            return MsConst.PYNATIVE_MODE

    @classmethod
    def start(cls, model=None):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if instance.task in PrecisionDebugger.task_not_need_service:
            return

        instance.config.execution_mode = cls._get_execution_mode()
        if cls._need_service():
            if not instance.service:
                instance.service = Service(instance.config)
            instance.service.start(model)
        else:
            if not instance.first_start:
                handler = TaskHandlerFactory.create(instance.config)
                handler.handle()

        instance.first_start = True
        Runtime.is_running = True

    @classmethod
    def forward_backward_dump_end(cls):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if instance.task in PrecisionDebugger.task_not_need_service:
            return
        if instance.service:
            instance.service.forward_backward_dump_end()

    @classmethod
    def stop(cls):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
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
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if instance.task in PrecisionDebugger.task_not_need_service:
            return
        if instance.service:
            instance.service.step()
        Runtime.step_count += 1

    @classmethod
    def monitor(cls, opt):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if instance.task != Const.GRAD_PROBE:
            return
        instance.gm.monitor(opt)

    @classmethod
    def _need_service(cls):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if instance.config.execution_mode != MsConst.PYNATIVE_MODE:
            return False
        else:
            return instance.config.task != Const.FREE_BENCHMARK and instance.config.level != MsConst.KERNEL
