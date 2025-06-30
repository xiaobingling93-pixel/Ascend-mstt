# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

from torch.utils.data import dataloader

from msprobe.core.common.const import Const, MsgConst
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.utils import check_token_range, ThreadSafe
from msprobe.core.debugger.precision_debugger import BasePrecisionDebugger
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import check_save_param, is_torch_nn_module
from msprobe.pytorch.debugger.debugger_config import DebuggerConfig
from msprobe.pytorch.dump.module_dump.module_dump import ModuleDumper
from msprobe.pytorch.grad_probe.grad_monitor import GradientMonitor
from msprobe.pytorch.pytorch_service import PytorchService
from msprobe.pytorch.pt_config import parse_task_config


class PrecisionDebugger(BasePrecisionDebugger):

    def __init__(
        self,
        config_path=None,
        task=None,
        dump_path=None,
        level=None,
        model=None,
        step=None
    ):
        if self.initialized:
            return
        super().__init__(config_path, task, dump_path, level, step)
        self.model = model
        if self.task == Const.GRAD_PROBE:
            self.gm = GradientMonitor(self.common_config, self.task_config)
            return
        self.config = DebuggerConfig(
            self.common_config, self.task_config, task, dump_path, level
        )
        self.service = PytorchService(self.config)
        self.module_dumper = ModuleDumper(self.service)
        self.ori_customer_func = {}
        self.enable_dataloader = self.config.enable_dataloader
        self._param_warning()

    @staticmethod
    def _get_task_config(task, json_config):
        return parse_task_config(task, json_config)

    @staticmethod
    def _iter_tracer(func):
        def func_wrapper(*args, **kwargs):
            debugger_instance = PrecisionDebugger._instance
            if not debugger_instance:
                raise MsprobeException(
                    MsprobeException.INTERFACE_USAGE_ERROR,
                    f"PrecisionDebugger must be instantiated before executing the dataloader iteration"
                )

            debugger_instance.enable_dataloader = False
            if not debugger_instance.service.first_start:
                debugger_instance.stop()
                debugger_instance.step()
            result = func(*args, **kwargs)
            debugger_instance.start()
            debugger_instance.enable_dataloader = True
            return result

        return func_wrapper

    @classmethod
    @ThreadSafe.synchronized
    def start(cls, model=None, token_range=None):
        instance = cls._get_instance()
        if instance is None:
            return

        check_token_range(token_range)
        instance.config.check_model(instance, model, token_range)

        if instance.enable_dataloader:
            logger.warning_on_rank_0("DataLoader is enabled, start() skipped.")
        else:
            instance.service.start(instance.model, token_range)

    @classmethod
    @ThreadSafe.synchronized
    def stop(cls):
        instance = cls._get_instance()
        if instance is None:
            return
        if instance.enable_dataloader:
            logger.warning_on_rank_0("DataLoader is enabled, stop() skipped.")
        else:
            instance.service.stop()

    @classmethod
    @ThreadSafe.synchronized
    def step(cls):
        instance = cls._get_instance()
        if instance is None:
            return
        cls._instance.service.step()

    @classmethod
    @ThreadSafe.synchronized
    def monitor(cls, model):
        if not cls._instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if cls._instance.task != Const.GRAD_PROBE:
            return
        cls._instance.gm.monitor(model)

    @classmethod
    @ThreadSafe.synchronized
    def save(cls, variable, name, save_backward=True):
        instance = cls._instance
        if not instance:
            raise Exception(MsgConst.NOT_CREATED_INSTANCE)
        if instance.task not in [Const.TENSOR, Const.STATISTICS] or instance.config.level != Const.LEVEL_DEBUG:
            return
        try:
            check_save_param(variable, name, save_backward)
        except ValueError:
            return
        instance.service.save(variable, name, save_backward)

    def _param_warning(self):
        if self.model is not None:
            logger.warning_on_rank_0(
                "The 'model' parameter in the PrecisionDebugger will be deprecated in the future."
                "It is recommended to pass the 'model' parameter in the start interface instead."
            )
        if self.enable_dataloader:
            logger.warning_on_rank_0("The enable_dataloader feature will be deprecated in the future.")
            dataloader._BaseDataLoaderIter.__next__ = self._iter_tracer(dataloader._BaseDataLoaderIter.__next__)


@ThreadSafe.synchronized
def module_dump(module, dump_name):
    if not is_torch_nn_module(module):
        raise MsprobeException(
            MsprobeException.INVALID_PARAM_ERROR,
            f"the module argument in module_dump must be a torch.nn.Module type, "
            f"but currently there is an unsupported {type(module)} type."
        )
    if not isinstance(dump_name, str):
        raise MsprobeException(
            MsprobeException.INVALID_PARAM_ERROR,
            f"the dump_name argument in module_dump must be a str type"
        )
    instance = PrecisionDebugger._instance
    if not instance:
        raise MsprobeException(
            MsprobeException.INTERFACE_USAGE_ERROR,
            f"PrecisionDebugger must be instantiated before using module_dump interface"
        )
    instance.module_dumper.start_module_dump(module, dump_name)


@ThreadSafe.synchronized
def module_dump_end():
    instance = PrecisionDebugger._instance
    if not instance:
        raise MsprobeException(
            MsprobeException.INTERFACE_USAGE_ERROR,
            f"PrecisionDebugger must be instantiated before using module_dump_end interface"
        )
    instance.module_dumper.stop_module_dump()
