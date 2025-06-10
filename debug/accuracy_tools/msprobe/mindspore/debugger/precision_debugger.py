# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
from collections import defaultdict, namedtuple

import mindspore as ms
from mindspore._c_expression import MSContext

from msprobe.core.common.const import Const, MsgConst
from msprobe.core.common.utils import check_token_range
from msprobe.core.common.runtime import Runtime
from msprobe.core.debugger.precision_debugger import BasePrecisionDebugger
from msprobe.mindspore.cell_processor import CellProcessor
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.common.utils import (
    set_register_backward_hook_functions,
    check_save_param,
    is_graph_mode_cell_dump_allowed
)
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.graph_mode_cell_dump import GraphModeCellDump
from msprobe.mindspore.dump.hook_cell.api_register import get_api_register
from msprobe.mindspore.dump.hook_cell.hook_cell import HOOKCell
from msprobe.mindspore.grad_probe.grad_monitor import GradientMonitor
from msprobe.mindspore.ms_config import parse_task_config
from msprobe.mindspore.mindspore_service import MindsporeService
from msprobe.mindspore.task_handler_factory import TaskHandlerFactory

try:
    from mindspore._c_expression import _dump_start, _dump_stop, _dump_step, _set_init_iter, _dump_set_dynamic
except ImportError:
    enable_dynamic_kbyk_dump = False
else:
    enable_dynamic_kbyk_dump = True

try:
    from msprobe.lib import _msprobe_c
except ImportError:
    _msprobe_c = None


ConfigParameters = namedtuple("ConfigParameters", ["config_path", "task", "dump_path", "level"])


class PrecisionDebugger(BasePrecisionDebugger):

    def __new__(cls, config_path=None, task=None, dump_path=None,
                level=None, step=None, opt=None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
            cls._instance.config = None
            cls.service = None
            cls.first_start = False
        return cls._instance

    def __init__(self, config_path=None, task=None, dump_path=None,
                 level=None, step=None):
        if self.initialized:
            return
        set_register_backward_hook_functions()
        super().__init__(config_path, task, dump_path, level, step)

        if self.task == Const.GRAD_PROBE:
            self.gm = GradientMonitor(self.common_config, self.task_config)
            return
        self.common_config.level = level if level else self.common_config.level
        self.common_config.dump_path = dump_path if dump_path else self.common_config.dump_path
        self.config = DebuggerConfig(self.common_config, self.task_config)

        if self._need_msprobe_c() and _msprobe_c:
            os.environ["MS_HOOK_ENABLE"] = "on"
            _msprobe_c._PrecisionDebugger(framework="MindSpore", config_path=config_path)

        self.config.execution_mode = self._get_execution_mode()
        if self._need_service():
            self.config.check_config_with_l2()
            self.service = MindsporeService(self.config)

        Runtime.step_count = 0
        Runtime.is_running = False
        if enable_dynamic_kbyk_dump:
            _dump_set_dynamic()

    @staticmethod
    def get_task_config(task, json_config):
        return parse_task_config(task, json_config)

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

    @staticmethod
    def _is_graph_dump(config: DebuggerConfig):
        if config.level != MsConst.KERNEL:
            return False
        if not config.list:
            return True
        is_graph = any(item.startswith("name-regex") for item in config.list)
        is_graph |= all("." not in item for item in config.list)
        return is_graph

    @classmethod
    def start(cls, model=None, token_range=None):
        instance = cls.get_instance()
        if instance is None:
            return
        if cls._need_msprobe_c() and _msprobe_c:
            _msprobe_c._PrecisionDebugger().start()
        check_token_range(token_range)
        instance.config.execution_mode = cls._get_execution_mode()
        if cls._need_service():
            if not instance.service:
                instance.service = MindsporeService(instance.config)
            instance.config.check_model(model, token_range)
            instance.service.start(model, token_range)
        else:
            if not instance.first_start:
                get_api_register().restore_all_api()
                handler = TaskHandlerFactory.create(instance.config, model)
                handler.handle()
                if enable_dynamic_kbyk_dump:
                    _set_init_iter(0)
            if enable_dynamic_kbyk_dump:
                is_valid_rank = (not instance.config.rank or Runtime.rank_id in instance.config.rank)
                is_valid_step = (not instance.config.step or Runtime.step_count in instance.config.step)
                if is_valid_rank and is_valid_step:
                    _dump_start()
            Runtime.is_running = True
        instance.first_start = True

    @classmethod
    def stop(cls):
        instance = cls.get_instance()
        if instance is None:
            return

        if instance.task == Const.GRAD_PROBE:
            instance.gm.stop()
        if instance.service:
            instance.service.stop()
        else:
            Runtime.is_running = False
        if enable_dynamic_kbyk_dump:
            _dump_stop()
        if cls._need_msprobe_c() and _msprobe_c:
            _msprobe_c._PrecisionDebugger().stop()
    
    @classmethod
    def step(cls):
        instance = cls.get_instance()
        if instance is None:
            return

        if instance.service:
            instance.service.step()
        if is_graph_mode_cell_dump_allowed(instance.config):
            GraphModeCellDump.step()
        if enable_dynamic_kbyk_dump:
            _dump_step(1)
        if cls._need_msprobe_c() and _msprobe_c:
            _msprobe_c._PrecisionDebugger().step()

        HOOKCell.cell_count = defaultdict(int)
        CellProcessor.reset_cell_stats()
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
    def save(cls, variable, name, save_backward=True):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if instance.task not in [Const.TENSOR, Const.STATISTICS] or instance.config.level_ori != Const.LEVEL_DEBUG:
            return
        try:
            check_save_param(variable, name, save_backward)
        except ValueError:
            return

        instance.config.execution_mode = cls._get_execution_mode()
        if cls._need_service():
            if not instance.service:
                instance.service = MindsporeService(instance.config)
            instance.service.save(variable, name, save_backward)

    @classmethod
    def _need_service(cls):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if instance.config.execution_mode != MsConst.PYNATIVE_MODE:
            return False
        else:
            return instance.config.task != Const.FREE_BENCHMARK and not instance._is_graph_dump(instance.config)

    @classmethod
    def _need_msprobe_c(cls):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        return instance.config.level_ori == Const.LEVEL_L2
