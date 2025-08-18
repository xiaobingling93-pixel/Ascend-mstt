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

import os
import threading
from abc import ABC, abstractmethod
from collections import defaultdict

from msprobe.core.common.runtime import Runtime
from msprobe.core.common.utils import Const, ThreadSafe
from msprobe.core.data_dump.data_processor.base import (ModuleBackwardInputsOutputs, ModuleForwardInputsOutputs)


class HookSet:
    def __init__(
            self,
            forward_pre_hook=None,
            forward_hook=None,
            backward_pre_hook=None,
            backward_hook=None,
            distributed_forward_hook=None
    ):
        self.forward_pre_hook = forward_pre_hook
        self.forward_hook = forward_hook
        self.backward_pre_hook = backward_pre_hook
        self.backward_hook = backward_hook
        self.distributed_forward_hook = distributed_forward_hook


class BaseHookManager(ABC):
    inner_switch = defaultdict(bool)
    inner_api_count = defaultdict(int)
    hook_handle_dict = {}
    params_grad_info = {}

    def __init__(self, data_collector, config):
        self.data_collector = data_collector
        self.config = config

    @property
    def _pid(self):
        return os.getpid()

    @property
    @abstractmethod
    def _is_recompute(self):
        pass

    @staticmethod
    def reset_status():
        BaseHookManager.inner_switch = defaultdict(bool)
        BaseHookManager.inner_api_count = defaultdict(int)
        BaseHookManager.hook_handle_dict.clear()
        BaseHookManager.params_grad_info.clear()

    @staticmethod
    def _clear_input_kwargs(module, tid):
        if hasattr(module, 'msprobe_input_kwargs') and tid in module.msprobe_input_kwargs:
            del module.msprobe_input_kwargs[tid]

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
    def _get_count(name):
        pass

    @staticmethod
    @abstractmethod
    def _process_kwargs_and_output(module, tid, hook_type, kwargs_or_output, output_or_kwargs):
        pass

    @abstractmethod
    def build_hook(self):
        pass

    @abstractmethod
    def _register_forward_hook(self, module, api_name):
        pass

    @abstractmethod
    def _register_backward_hook(self, module, full_backward_name, args):
        pass

    @abstractmethod
    def _register_backward_pre_hook(self, module, full_backward_name, output):
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
                    handle = param.register_hook(self._build_grad_hook(ori_name, param_name))
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

    def _should_execute_hook(self, hook_type, tid, is_forward=True):
        is_api_hook = hook_type == Const.API
        if BaseHookManager.inner_switch[tid]:
            return False
        if not is_api_hook and not Runtime.is_running:
            return False
        elif is_api_hook and is_forward and not Runtime.is_running:
            return False
        if not self.data_collector or self.data_collector.data_processor.is_terminated:
            return False
        return True

    def _build_grad_hook(self, ori_name, param_name):
        def hook_fn(grad):
            tid = threading.get_ident()
            if not self._should_execute_hook(Const.MODULE, tid):
                return
            with ThreadSafe():
                BaseHookManager.inner_switch[tid] = True
                self.data_collector.params_data_collect(ori_name, param_name, self._pid, grad)
                BaseHookManager.inner_switch[tid] = False
            return

        return hook_fn

    def _build_forward_pre_hook(self, hook_type, api_name):
        def forward_pre_hook(module, args, kwargs=None):
            if hook_type == Const.MODULE:
                return None

            tid = threading.get_ident()
            if not self._should_execute_hook(hook_type, tid):
                return None

            with ThreadSafe():
                self._register_forward_hook(module, api_name)
                BaseHookManager.inner_api_count[tid] += 1
                if BaseHookManager.inner_api_count[tid] != 1:
                    return None

                full_forward_name = api_name + str(self._get_count(api_name)) + Const.SEP + Const.FORWARD
                full_backward_name = api_name + str(self._get_count(api_name)) + Const.SEP + Const.BACKWARD
                module.full_forward_name = full_forward_name
                if kwargs is None:
                    kwargs = module.msprobe_input_kwargs.get(tid, {}) if hasattr(module, 'msprobe_input_kwargs') else {}
                BaseHookManager.inner_switch[tid] = True
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=None)

                args = self._register_backward_hook(module, full_backward_name, args)
                with self._no_grad_context():
                    self.data_collector.update_api_or_module_name(full_forward_name)
                    self.data_collector.forward_input_data_collect(
                        full_forward_name,
                        module,
                        self._pid,
                        module_input_output,
                        self._is_recompute
                    )
                BaseHookManager.inner_switch[tid] = False
                return args

        return forward_pre_hook

    def _build_forward_hook(self, hook_type, api_name):
        def forward_hook(module, args, kwargs_or_output, output_or_kwargs=None):
            tid = threading.get_ident()
            if not self._should_execute_hook(hook_type, tid):
                self._clear_input_kwargs(module, tid)
                return None

            with ThreadSafe():
                if hook_type == Const.API:
                    if BaseHookManager.inner_api_count[tid] != 1:
                        if BaseHookManager.inner_api_count[tid] > 1:
                            BaseHookManager.inner_api_count[tid] -= 1
                        self._clear_input_kwargs(module, tid)
                        return None

                kwargs, output = self._process_kwargs_and_output(
                    module,
                    tid,
                    hook_type,
                    kwargs_or_output,
                    output_or_kwargs
                )
                BaseHookManager.inner_switch[tid] = True
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=output)
                if hook_type == Const.API:
                    full_forward_name = api_name + str(self._get_count(api_name)) + Const.SEP + Const.FORWARD
                    full_backward_name = api_name + str(self._get_count(api_name)) + Const.SEP + Const.BACKWARD
                    output = self._register_backward_pre_hook(module, full_backward_name, output)

                with self._no_grad_context():
                    if hook_type == Const.MODULE:
                        params_dict = self._get_params_dict(module)
                        setattr(module_input_output, Const.PARAMS, params_dict)
                        if params_dict:
                            self._register_param_hook(api_name, module, params_dict)
                        self.data_collector.update_api_or_module_name(api_name)
                        self.data_collector.forward_data_collect(
                            api_name,
                            module,
                            self._pid,
                            module_input_output,
                            self._is_recompute
                        )
                        self._init_params_grad_info(module, params_dict)
                    else:
                        self.data_collector.update_api_or_module_name(full_forward_name)
                        self.data_collector.forward_output_data_collect(
                            full_forward_name,
                            module,
                            self._pid,
                            module_input_output,
                            self._is_recompute
                        )
                        self._add_count(api_name)
                        BaseHookManager.inner_api_count[tid] -= 1
                    self._clear_input_kwargs(module, tid)

                    if self.data_collector.if_return_forward_new_output():
                        forward_new_output = self.data_collector.get_forward_new_output()
                        BaseHookManager.inner_switch[tid] = False
                        return forward_new_output

                BaseHookManager.inner_switch[tid] = False
                return output

        return forward_hook

    def _build_backward_hook(self, hook_type, full_name):
        def backward_hook(module, grad_input, grad_output):
            tid = threading.get_ident()
            if not self._should_execute_hook(hook_type, tid, is_forward=False):
                return

            with ThreadSafe():
                BaseHookManager.inner_switch[tid] = True
                self.data_collector.update_api_or_module_name(full_name)

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
                BaseHookManager.inner_switch[tid] = False

        return backward_hook
