# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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

from msprobe.core.common.utils import Const
from msprobe.core.service import BaseService
from msprobe.pytorch.attl_manager import ATTLManager
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import get_rank_if_initialized, torch_version_above_or_equal_2
from msprobe.pytorch.dump.module_dump.module_processer import ModuleProcesser
from msprobe.pytorch.hook_module.api_register import get_api_register, ApiTemplate
from msprobe.pytorch.hook_module.hook_module import HOOKModule
from msprobe.pytorch.hook_module.jit_script_wrapper import wrap_jit_script_func
from msprobe.pytorch.hook_module.pt_hook_manager import PytorchHookManager
from msprobe.pytorch.hook_module.register_optimizer_hook import register_optimizer_hook

if torch_version_above_or_equal_2:
    from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.dump_dispatch import run_ut_dispatch


class PytorchService(BaseService):
    @property
    def _get_framework_type(self):
        return Const.PT_FRAMEWORK

    @staticmethod
    def _get_current_rank():
        return get_rank_if_initialized()

    def _init_specific_components(self):
        self.logger = logger
        self.api_register = get_api_register()
        self.module_processor = ModuleProcesser(self.data_collector.scope)
        self.attl_manager = ATTLManager(self.config)
        self.hook_manager = PytorchHookManager(self.data_collector, self.config, self.attl_manager)
        self.api_template = ApiTemplate

    def _register_hook(self):
        self.attl_manager.attl_init()
        if self._is_mix_level:
            register_optimizer_hook(self.data_collector)

    def _register_api_hook(self):
        super()._register_api_hook()
        wrap_jit_script_func()

    def _register_module_hook(self):
        ModuleProcesser.enable_module_dump = True
        self.module_processor.register_module_hook(self.model, self.build_hook)
        self.logger.info(f"The module {self.config.task} hook function is successfully mounted to the model.")

    def _run_ut_dispatch(self, status):
        if torch_version_above_or_equal_2:
            run_ut_dispatch(self.attl_manager.attl, status, self.config.online_run_ut_recompute)

    def _reset_status(self):
        super()._reset_status()
        ModuleProcesser.reset_module_stats()
        HOOKModule.reset_module_stats()
