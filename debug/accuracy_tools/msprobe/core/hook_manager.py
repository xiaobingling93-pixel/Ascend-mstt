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


from abc import ABC, abstractmethod
import os

from msprobe.core.common.runtime import Runtime
from msprobe.core.common.utils import Const
from msprobe.core.data_dump.data_processor.base import (ModuleBackwardInputsOutputs, ModuleForwardInputsOutputs)


class HookSet:
    def __init__(self, forward_hook=None, forward_pre_hook=None, backward_hook=None, backward_pre_hook=None):
        self.forward_hook = forward_hook
        self.forward_pre_hook = forward_pre_hook
        self.backward_hook = backward_hook
        self.backward_pre_hook = backward_pre_hook


class BaseHookManager(ABC):
    inner_switch = False
    hook_handle_dict = {}
    params_grad_info = {}

    def __init__(self, data_collector, config, attl_manager=None):
        self.data_collector = data_collector
        self.config = config
        self.attl_manager = attl_manager

    @property
    def _pid(self):
        return os.getpid()

    @property
    @abstractmethod
    def _is_recompute(self):
        pass

    @staticmethod
    @abstractmethod
    def _no_grad_context():
        pass

    @staticmethod
    @abstractmethod
    def _add_count(name):
        pass

    @staticmethod
    @abstractmethod
    def _process_kwargs_and_output(module, hook_type, kwargs_or_output, output_or_kwargs):
        pass

    @staticmethod
    def _clear_input_kwargs(module):
        if hasattr(module, 'msprobe_input_kwargs'):
            del module.msprobe_input_kwargs

    @abstractmethod
    def build_hook(self):
        pass

    @abstractmethod
    def _get_params_dict(self, module):
        pass

    @abstractmethod
    def _need_exchange(self, module):
        pass

    def _register_param_hook(self, name, module, params_dict):
        ori_name = name.rsplit(Const.SEP, 2)[0]
        grad_name = ori_name + Const.SEP + Const.PARAMS_GRAD
        # 首次执行前向hook时，添加params_grad_name属性，并注册参数hook
        setattr(module, 'params_grad_name', grad_name)
         # data_mode为forward时，不注册参数hook
        if not (Const.FORWARD in self.config.data_mode and Const.BACKWARD not in self.config.data_mode):
            for param_name, param in params_dict.items():
                if param.requires_grad:
                    name = ori_name + Const.SEP + param_name
                    old_handle = BaseHookManager.hook_handle_dict.get(name)
                    if old_handle and hasattr(old_handle, "remove"):
                        old_handle.remove()
                    handle = param.register_hook(self._build_grad_hook(module, ori_name, param_name))
                    BaseHookManager.hook_handle_dict[name] = handle

    def _init_params_grad_info(self, module, params_dict):
        '''
        初始化参数梯度信息, 在前向hook结束后, 将参数梯度信息写入cache_data中用于占位
        '''
        if not params_dict:
            return
        if not (Const.FORWARD in self.config.data_mode and Const.BACKWARD not in self.config.data_mode):
            grad_name = module.params_grad_name if hasattr(module, 'params_grad_name') else None
            # 判断是否已经在cache_data中进行了占位, 若没有则先写入cache_data中
            if not BaseHookManager.params_grad_info.get(grad_name):
                data_info = {grad_name: {key: [None] for key, value in params_dict.items() if value.requires_grad}}
                # 当模块中的参数有requires_grad属性为True时，才会进行梯度计算，此时才需要占位
                if data_info.get(grad_name):
                    # 将grad_name的data_info先写入cache_data中, 梯度计算后再更新
                    self.data_collector.handle_data(grad_name, data_info,
                                                    flush=self.data_collector.data_processor.is_terminated)
                # 记录当前模块的参数梯度信息已占位
                BaseHookManager.params_grad_info[grad_name] = True

    def _should_execute_hook(self, hook_type, module, is_forward):
        is_module_hook = hook_type == Const.MODULE
        if is_module_hook and not Runtime.is_running:
            return False
        elif not is_module_hook and is_forward and not Runtime.is_running:
            return False
        elif not is_module_hook and not is_forward and not module.forward_data_collected:
            return False
        if BaseHookManager.inner_switch:
            return False
        if not self.data_collector or self.data_collector.data_processor.is_terminated:
            return False
        return True

    def _build_grad_hook(self, module, ori_name, param_name):
        def hook_fn(grad):
            if not self._should_execute_hook(Const.MODULE, module, False):
                return
            BaseHookManager.inner_switch = True
            self.data_collector.params_data_collect(ori_name, param_name, self._pid, grad)
            BaseHookManager.inner_switch = False
            return
        return hook_fn

    def _build_forward_pre_hook(self, hook_type, full_name, api_name):
        def forward_pre_hook(module, args, kwargs=None):
            if hook_type == Const.MODULE:
                return
            if not self._should_execute_hook(hook_type, module, True):
                return
            if kwargs is None:
                kwargs = module.msprobe_input_kwargs if hasattr(module, 'msprobe_input_kwargs') else {}
            with self._no_grad_context():
                BaseHookManager.inner_switch = False
                module.forward_data_collected = True
                self._add_count(api_name)
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=None)
                self.data_collector.update_api_or_module_name(full_name)
                if getattr(self.config, "online_run_ut", False):
                    BaseHookManager.inner_switch = False
                    return
                self.data_collector.forward_input_data_collect(
                    full_name,
                    module,
                    self._pid,
                    module_input_output,
                    self._is_recompute
                )
                BaseHookManager.inner_switch = False
        return forward_pre_hook

    def _build_forward_hook(self, hook_type, full_name):
        def forward_hook(module, args, kwargs_or_output, output_or_kwargs=None):
            if not self._should_execute_hook(hook_type, module, True):
                self._clear_input_kwargs(module)
                return None
            kwargs, output = self._process_kwargs_and_output(module, hook_type, kwargs_or_output, output_or_kwargs)
            BaseHookManager.inner_switch = True
            self.data_collector.update_api_or_module_name(full_name)
            module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=output)
            with self._no_grad_context():
                if getattr(self.config, "online_run_ut", False):
                    if self.data_collector.scope and not self.data_collector.scope.check(full_name):
                        return None
                    if self.attl_manager:
                        self.attl_manager.attl_send(full_name, args, kwargs, output)
                    BaseHookManager.inner_switch = False
                    return None
                if hook_type == Const.MODULE:
                    params_dict = self._get_params_dict(module)
                    setattr(module_input_output, Const.PARAMS, params_dict)
                    if params_dict:
                        self._register_param_hook(full_name, module, params_dict)
                    self.data_collector.update_api_or_module_name(full_name)
                    self.data_collector.forward_data_collect(
                        full_name,
                        module,
                        self._pid,
                        module_input_output,
                        self._is_recompute
                    )
                    self._init_params_grad_info(module, params_dict)
                else:
                    self.data_collector.forward_output_data_collect(
                        full_name,
                        module,
                        self._pid,
                        module_input_output,
                        self._is_recompute
                    )
                self._clear_input_kwargs(module)

                if self.data_collector.if_return_forward_new_output():
                    forward_new_output = self.data_collector.get_forward_new_output()
                    BaseHookManager.inner_switch = False
                    return forward_new_output

                BaseHookManager.inner_switch = False
                return output
        return forward_hook

    def _build_backward_hook(self, hook_type, full_name):
        def backward_hook(module, grad_input, grad_output):
            if not self._should_execute_hook(hook_type, module, False):
                return
            BaseHookManager.inner_switch = True
            self.data_collector.update_api_or_module_name(full_name)
            if getattr(self.config, "online_run_ut", False):
                BaseHookManager.inner_switch = False
                return
            need_exchange = self._need_exchange(module) if hook_type == Const.MODULE else True
            if need_exchange:
                module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_output, grad_output=grad_input)
            else:
                module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_input, grad_output=grad_output)
            self.data_collector.backward_data_collect(
                    full_name,
                    module,
                    self._pid,
                    module_input_output,
                    self._is_recompute
                )
            BaseHookManager.inner_switch = False
        return backward_hook
