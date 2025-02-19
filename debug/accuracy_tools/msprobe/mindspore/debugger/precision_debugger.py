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

from msprobe.core.common.const import Const, FileCheckConst, MsgConst
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.file_utils import FileChecker
from msprobe.core.common.utils import get_real_step_or_rank
from msprobe.mindspore.cell_processor import CellProcessor
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.common.utils import set_register_backward_hook_functions, check_save_param
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.hook_cell.api_registry import api_register
from msprobe.mindspore.dump.hook_cell.hook_cell import HOOKCell
from msprobe.mindspore.grad_probe.grad_monitor import GradientMonitor
from msprobe.mindspore.ms_config import parse_json_config
from msprobe.mindspore.runtime import Runtime
from msprobe.mindspore.service import Service
from msprobe.mindspore.task_handler_factory import TaskHandlerFactory

try:
    from msprobe.lib import _msprobe_c
except ImportError:
    _msprobe_c = None


ConfigParameters = namedtuple("ConfigParameters", ["config_path", "task", "dump_path", "level"])


class PrecisionDebugger:
    _instance = None
    task_not_need_service = [Const.GRAD_PROBE]

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
        self.initialized = True

        set_register_backward_hook_functions()

        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), "../../config.json")

        config_params = ConfigParameters(config_path, task, dump_path, level)
        self.check_input_params(config_params)

        common_config, task_config = parse_json_config(config_path)
        common_config.task = task if task else common_config.task
        self.task = common_config.task
        if self.task == Const.GRAD_PROBE:
            self.gm = GradientMonitor(common_config, task_config)
            return
        common_config.step = get_real_step_or_rank(
            step, Const.STEP) if step is not None else common_config.step
        common_config.level = level if level else common_config.level
        common_config.dump_path = dump_path if dump_path else common_config.dump_path
        self.config = DebuggerConfig(common_config, task_config)

        if _msprobe_c:
            _msprobe_c._PrecisionDebugger(framework="MindSpore", config_path=config_path)

        self.config.execution_mode = self._get_execution_mode()
        if self._need_service():
            self.config.check_config_with_l2()
            self.service = Service(self.config)

        Runtime.step_count = 0
        Runtime.is_running = False

    @staticmethod
    def check_input_params(args):
        if args.config_path is not None:
            if not isinstance(args.config_path, str):
                raise MsprobeException(
                    MsprobeException.INVALID_PARAM_ERROR, f"config_path must be a string")
            file_checker = FileChecker(
                file_path=args.config_path, path_type=FileCheckConst.FILE, file_type=FileCheckConst.JSON_SUFFIX)
            file_checker.common_check()

        if args.task is not None and args.task not in Const.TASK_LIST:
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, f"task must be one of {Const.TASK_LIST}")

        if args.dump_path is not None:
            if not isinstance(args.dump_path, str):
                raise MsprobeException(
                    MsprobeException.INVALID_PARAM_ERROR, f"dump_path must be a string")

        if args.level is not None and args.level not in Const.LEVEL_LIST:
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, f"level must be one of {Const.LEVEL_LIST}")

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
    def _is_graph_dump(config):
        if config.level != MsConst.KERNEL:
            return False
        if not config.list:
            return True
        is_graph = any(item.startswith("name-regex") for item in config.list)
        is_graph |= all("." not in item for item in config.list)
        return is_graph

    @classmethod
    def start(cls, model=None):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if _msprobe_c:
            _msprobe_c._PrecisionDebugger().start()
        if instance.task in PrecisionDebugger.task_not_need_service:
            return

        instance.config.execution_mode = cls._get_execution_mode()
        if cls._need_service():
            if not instance.service:
                instance.service = Service(instance.config)
            instance.service.start(model)
        else:
            if not instance.first_start:
                api_register.api_set_ori_func()
                handler = TaskHandlerFactory.create(instance.config)
                handler.handle()

        instance.first_start = True
        Runtime.is_running = True

    @classmethod
    def forward_backward_dump_end(cls):
        instance = cls._instance
        instance.stop()

    @classmethod
    def stop(cls):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if _msprobe_c:
            _msprobe_c._PrecisionDebugger().stop()
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
        if _msprobe_c:
            _msprobe_c._PrecisionDebugger().step()
        if instance.task in PrecisionDebugger.task_not_need_service:
            return
        if instance.service:
            instance.service.step()
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
                instance.service = Service(instance.config)
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