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


from contextlib import nullcontext

from msprobe.core.common.const import Const
from msprobe.core.common.utils import replace_last_occurrence
from msprobe.core.hook_manager import BaseHookManager, HookSet
from msprobe.pytorch.common.utils import is_recomputation, torch_version_above_or_equal_2
from msprobe.pytorch.hook_module.hook_module import HOOKModule


class PytorchHookManager(BaseHookManager): 
    @property
    def _is_recompute(self):
        return is_recomputation()

    @staticmethod
    def _no_grad_context():
        return nullcontext()

    @staticmethod
    def _add_count(name):
        HOOKModule.add_module_count(name)

    @staticmethod
    def _process_kwargs_and_output(module, hook_type, kwargs_or_output, output_or_kwargs):
        kwargs = kwargs_or_output if torch_version_above_or_equal_2 else {}
        output = output_or_kwargs if torch_version_above_or_equal_2 else kwargs_or_output
        return kwargs, output
    
    def build_hook(self, hook_type, name):
        if hook_type == Const.API:
            full_forward_name = name + str(HOOKModule.get_module_count(name)) + Const.SEP + Const.FORWARD
        else:
            full_forward_name = name
        full_backward_name = replace_last_occurrence(full_forward_name, Const.FORWARD, Const.BACKWARD)
        hookset = HookSet(
            forward_hook=self._build_forward_hook(hook_type, full_forward_name),
            forward_pre_hook=self._build_forward_pre_hook(hook_type, full_forward_name, name),
            backward_hook=self._build_backward_hook(hook_type, full_backward_name)        
        )
        return hookset
    
    def _need_exchange(self, module):
        return True

    def _get_params_dict(self, module):
        params_dict = {}
        if self.config.task != Const.STRUCTURE:
            params_dict = {
                    key.split(Const.SEP)[-1]: value
                    for key, value in module.named_parameters(recurse=False)
                    }
        return params_dict
