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

from collections import namedtuple

import torch
from msprobe.core.common.const import Const, FileCheckConst, MsgConst
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.file_utils import FileChecker
from msprobe.core.common.utils import get_real_step_or_rank
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.debugger.debugger_config import DebuggerConfig
from msprobe.pytorch.grad_probe.grad_monitor import GradientMonitor
from msprobe.pytorch.pt_config import parse_json_config
from msprobe.pytorch.service import Service
from torch.utils.data import dataloader

ConfigParameters = namedtuple("ConfigParameters", ["config_path", "task",
                                                   "dump_path", "level", "model"])


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
            config_params = ConfigParameters(config_path,
                                             task,
                                             dump_path,
                                             level,
                                             model)
            self.check_input_params(config_params)

            self.api_origin = False
            self.initialized = True
            self.model = model
            common_config, task_config = parse_json_config(config_path, task)
            self.task = task if task else common_config.task
            if self.task == Const.GRAD_PROBE:
                self.gm = GradientMonitor(common_config, task_config)
                return
            if step:
                common_config.step = get_real_step_or_rank(step, Const.STEP)
            self.config = DebuggerConfig(
                common_config, task_config, task, dump_path, level
            )
            self.service = Service(self.config)
            self.enable_dataloader = self.config.enable_dataloader
            if self.enable_dataloader:
                logger.warning_on_rank_0("The enable_dataloader feature will be deprecated in the future.")
                dataloader._BaseDataLoaderIter.__next__ = iter_tracer(dataloader._BaseDataLoaderIter.__next__)

    @property
    def instance(self):
        return self._instance

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

        if args.model is not None and not isinstance(args.model, torch.nn.Module):
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, f"model must be a torch.nn.Module")

    @classmethod
    def start(cls, model=None):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if instance.task in PrecisionDebugger.tasks_not_need_debugger:
            return
        instance.config.check_model(instance, model)
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
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if instance.task in PrecisionDebugger.tasks_not_need_debugger:
            return
        if instance.enable_dataloader:
            logger.warning_on_rank_0("DataLoader is enabled, stop() skipped.")
        else:
            instance.service.stop()

    @classmethod
    def step(cls):
        if not cls._instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if cls._instance.task in PrecisionDebugger.tasks_not_need_debugger:
            return
        cls._instance.service.step()

    @classmethod
    def monitor(cls, model):
        if not cls._instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
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
