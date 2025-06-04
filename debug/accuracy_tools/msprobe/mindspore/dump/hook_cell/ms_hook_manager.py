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

from mindspore.common.api import _no_grad
from msprobe.core.common.const import Const
from msprobe.core.common.utils import replace_last_occurrence
from msprobe.core.data_dump.data_processor.base import ModuleBackwardInputs
from msprobe.core.hook_manager import BaseHookManager, HookSet
from msprobe.mindspore.common.utils import has_kwargs_in_forward_hook
from msprobe.mindspore.dump.hook_cell.hook_cell import HOOKCell


class MindsproeHookManager(BaseHookManager):
    @property
    def _is_recompute(self):
        return None

    @staticmethod
    def _no_grad_context():
        return _no_grad()

    @staticmethod
    def _add_count(name):
        HOOKCell.add_cell_count(name)

    @staticmethod
    def _process_kwargs_and_output(module, hook_type, kwargs_or_output, output_or_kwargs):
        if not has_kwargs_in_forward_hook() or hook_type == Const.API:
            kwargs = module.msprobe_input_kwargs if hasattr(module, 'msprobe_input_kwargs') else {}
            output = kwargs_or_output
        else:
            kwargs = kwargs_or_output
            output = output_or_kwargs
        return kwargs, output

    def build_hook(self, hook_type, name):
        if hook_type == Const.API:
            full_forward_name = name + str(HOOKCell.get_cell_count(name)) + Const.SEP + Const.FORWARD
        else:
            full_forward_name = name
        full_backward_name = replace_last_occurrence(full_forward_name, Const.FORWARD, Const.BACKWARD)
        hookset = HookSet(
            forward_hook=self._build_forward_hook(hook_type, full_forward_name),
            forward_pre_hook=self._build_forward_pre_hook(hook_type, full_forward_name, name),
            backward_hook=self._build_backward_hook(hook_type, full_backward_name),
            backward_pre_hook=self._build_backward_pre_hook(hook_type, full_backward_name)   
        )
        return hookset

    def _need_exchange(self, module):
        if not hasattr(module, 'has_pre_hook_called') or not module.has_pre_hook_called:
            return False
        else:
            return True

    def _get_params_dict(self, module):
        params_dict = {}
        if self.config.task != Const.STRUCTURE:
            params_dict = {
                    key.split(Const.SEP)[-1]: value
                    for key, value in module.parameters_dict(recurse=False).items()
                    }
        return params_dict

    def _build_backward_pre_hook(self, hook_type, name):
        def backward_pre_hook(module, grad_input):
            if self.config.level != Const.LEVEL_L2:
                return
            if not self._should_execute_hook(hook_type, module, False):
                return
            BaseHookManager.inner_switch = True
            module_input = ModuleBackwardInputs(grad_input=grad_input)
            self.data_collector.update_api_or_module_name(name)
            self.data_collector.backward_input_data_collect(name, module, self._pid, module_input)
            BaseHookManager.inner_switch = False
        return backward_pre_hook
