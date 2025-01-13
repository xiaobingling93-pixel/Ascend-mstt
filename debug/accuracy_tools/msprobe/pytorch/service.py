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

import functools
import os
from collections import namedtuple

import torch
from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import DistributedNotInitializedError
from msprobe.core.common.file_utils import create_directory
from msprobe.core.common.utils import print_tools_ends_info
from msprobe.core.data_dump.data_collector import build_data_collector
from msprobe.core.data_dump.data_processor.base import ModuleForwardInputsOutputs, ModuleBackwardInputsOutputs
from msprobe.core.data_dump.scope import BaseScope
from msprobe.pytorch.api_accuracy_checker.common.utils import ApiData
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import get_rank_if_initialized
from msprobe.pytorch.dump.kernel_dump.kernel_config import create_kernel_config_json
from msprobe.pytorch.dump.module_dump.module_processer import ModuleProcesser
from msprobe.pytorch.hook_module.api_registry import api_register
from msprobe.pytorch.hook_module.hook_module import HOOKModule

torch_version_above_or_equal_2 = torch.__version__.split('+')[0] >= '2.0'
if torch_version_above_or_equal_2:
    from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.dump_dispatch import run_ut_dispatch

HookFn = namedtuple('hookFn', ['pre_hook', 'forward_hook', 'backward_hook', 'forward_hook_torch_version_below_2'])


class Service:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.data_collector = build_data_collector(config)
        self.module_processor = ModuleProcesser(self.data_collector.scope)
        self.switch = False
        self.inner_switch = False
        self.current_iter = 0
        self.first_start = True
        self.current_rank = None
        self.dump_iter_dir = None
        self.should_stop_service = False
        self.attl = None

    def build_hook(self, module_type, name):
        def pre_hook(api_or_module_name, module, args, kwargs):
            if not self.should_execute_hook(module_type, module, True):
                return args, kwargs

            self.inner_switch = True
            if module_type == BaseScope.Module_Type_Module:
                api_or_module_name = module.mindstudio_reserved_name[-1]
            else:
                module.forward_data_collected = True
                HOOKModule.add_module_count(name)
            self.data_collector.update_api_or_module_name(api_or_module_name)

            if self.config.online_run_ut:
                self.inner_switch = False
                return None, None
            if self.data_collector:
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=None)
                self.data_collector.forward_input_data_collect(api_or_module_name, module, pid, module_input_output)

            self.inner_switch = False
            return args, kwargs

        def grad_hook(module, ori_name, param_name):
            def hook_fn(grad):
                if not self.should_execute_hook(module_type, module, False):
                    return grad
                self.inner_switch = True
                self.data_collector.params_data_collect(ori_name, param_name, pid, grad)
                self.inner_switch = False
                return grad

            return hook_fn

        def register_param_hook(module_name, module, params_dict):
            # data_mode为forward时，不注册参数hook
            if not (Const.FORWARD in self.config.data_mode and Const.BACKWARD not in self.config.data_mode):
                # 判断参数是否已经注册过hook
                if params_dict and hasattr(module, 'has_param_hook') and not module.has_param_hook:
                    ori_name = module_name.rsplit(Const.SEP, 2)[0]
                    grad_name = ori_name + Const.SEP + Const.PARAMS_GRAD
                    # 注册hook时，初始化grad_name的data_info
                    data_info = {grad_name: {key: [None] for key in params_dict}}
                    # 将grad_name的data_info先写入cache_data中, 梯度计算后再更新
                    self.data_collector.handle_data(grad_name, data_info,
                                                    flush=self.data_collector.data_processor.is_terminated)
                    for param_name, param in params_dict.items():
                        param.register_hook(grad_hook(module, ori_name, param_name))
                    module.has_param_hook = True

        def forward_hook(api_or_module_name, module, args, kwargs, output):
            if not self.should_execute_hook(module_type, module, True):
                return None

            self.inner_switch = True
            if self.config.online_run_ut:
                self.data_collector.update_api_or_module_name(api_or_module_name)
                if self.data_collector.scope and not self.data_collector.scope.check(api_or_module_name):
                    return None
                api_data = ApiData(
                    api_or_module_name[:-len(Const.FORWARD_NAME_SUFFIX)],
                    args,
                    kwargs,
                    output,
                    self.current_iter,
                    self.current_rank
                )
                self.attl_send(api_data)
                self.inner_switch = False
                return None

            module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=output)
            if module_type == BaseScope.Module_Type_Module:
                api_or_module_name = module.mindstudio_reserved_name[-1]
                self.data_collector.update_api_or_module_name(api_or_module_name)
                params_dict = {key.split(Const.SEP)[-1]: value for key, value in module.named_parameters(recurse=False)}
                setattr(module_input_output, Const.PARAMS, params_dict)
                # 设置has_param_hook属性，避免重复注册hook
                if not hasattr(module, 'has_param_hook'):
                    setattr(module, 'has_param_hook', False)
                self.data_collector.forward_data_collect(
                    api_or_module_name,
                    module,
                    pid,
                    module_input_output
                )
                register_param_hook(api_or_module_name, module, params_dict)
            else:
                self.data_collector.update_api_or_module_name(api_or_module_name)
                self.data_collector.forward_output_data_collect(
                    api_or_module_name,
                    module,
                    pid,
                    module_input_output
                )

            if self.data_collector.if_return_forward_new_output():
                forward_new_output = self.data_collector.get_forward_new_output()
                self.inner_switch = False
                return forward_new_output
            self.inner_switch = False
            return output

        def forward_hook_torch_version_below_2(api_or_module_name, module, args, output):
            return forward_hook(api_or_module_name, module, args, {}, output)

        def backward_hook(api_or_module_name, module, grad_input, grad_output):
            if not self.should_execute_hook(module_type, module, False):
                return

            self.inner_switch = True
            if module_type == BaseScope.Module_Type_Module:
                api_or_module_name = module.mindstudio_reserved_name[-1]
            self.data_collector.update_api_or_module_name(api_or_module_name)

            if self.config.online_run_ut:
                self.inner_switch = False
                return

            if self.data_collector:
                # 此处获取到的grad_input实际为反向过程的输出数据，grad_output为反向过程的输入数据，因此传入时调换顺序
                module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_output, grad_output=grad_input)
                self.data_collector.backward_data_collect(api_or_module_name, module, pid, module_input_output)
            self.inner_switch = False

        pid = os.getpid()
        full_forward_name = None
        full_backward_name = None
        if module_type == BaseScope.Module_Type_API:
            full_forward_name = name + str(HOOKModule.get_module_count(name)) + Const.SEP + Const.FORWARD
            full_backward_name = name + str(HOOKModule.get_module_count(name)) + Const.SEP + Const.BACKWARD
        pre_forward_hook_fn = functools.partial(pre_hook, full_forward_name)
        forward_hook_fn = functools.partial(forward_hook, full_forward_name)
        backward_hook_fn = functools.partial(backward_hook, full_backward_name)
        forward_hook_torch_version_below_2_fn = functools.partial(
            forward_hook_torch_version_below_2,
            full_forward_name
        )
        return HookFn(pre_forward_hook_fn, forward_hook_fn, backward_hook_fn, forward_hook_torch_version_below_2_fn)

    def start(self, model):
        if self.need_stop_service():
            return

        self.model = model
        if self.first_start:
            try:
                self.current_rank = get_rank_if_initialized()
            except DistributedNotInitializedError:
                self.current_rank = None
            self.attl_init()

            if self.config.rank and self.current_rank not in self.config.rank:
                return
            self.register_hook_new()
            self.first_start = False
        if self.config.online_run_ut and torch_version_above_or_equal_2:
            run_ut_dispatch(self.attl, True, self.config.online_run_ut_recompute)
        self.switch = True
        logger.info_on_rank_0(f"Dump switch is turned on at step {self.current_iter}. ")
        if not self.config.online_run_ut:
            self.create_dirs()
            logger.info_on_rank_0(f"Dump data will be saved in {self.dump_iter_dir}.")

    def stop(self):
        if self.should_stop_service:
            return
        if self.config.step and self.current_iter not in self.config.step:
            return
        if self.config.rank and self.current_rank not in self.config.rank:
            return
        self.switch = False
        if self.config.level == Const.LEVEL_L2:
            return
        if self.config.online_run_ut and torch_version_above_or_equal_2:
            run_ut_dispatch(self.attl, False, self.config.online_run_ut_recompute)
            return
        self.data_collector.write_json()

    def step(self):
        if self.should_stop_service:
            return
        self.data_collector.write_json()
        self.current_iter += 1
        self.data_collector.update_iter(self.current_iter)
        self.reset_status()

    def need_stop_service(self):
        if self.should_stop_service:
            return True
        end_service = self.config.step and self.current_iter > max(self.config.step) or \
                      self.data_collector and self.data_collector.data_processor.is_terminated
        if end_service:
            if self.config.online_run_ut:
                # send stop signal if online_run_ut
                self.attl_stop()
            if self.config.level in [Const.LEVEL_L1, Const.LEVEL_L2, Const.LEVEL_MIX]:
                api_register.api_originality()
            self.switch = False
            self.should_stop_service = True
            print_tools_ends_info()
            return True
        if self.config.step and self.current_iter not in self.config.step:
            return True
        return False

    def should_execute_hook(self, hook_type, module, is_forward):
        is_module_hook = hook_type == BaseScope.Module_Type_Module
        if is_module_hook and not self.switch:
            return False
        elif not is_module_hook and is_forward and not self.switch:
            return False
        elif not is_module_hook and not is_forward and not module.forward_data_collected:
            return False

        if self.inner_switch:
            return False
        if not self.data_collector or self.data_collector.data_processor.is_terminated:
            return False
        return True

    def create_dirs(self):
        create_directory(self.config.dump_path)
        self.dump_iter_dir = os.path.join(self.config.dump_path, f"step{self.current_iter}")
        cur_rank = self.current_rank if self.current_rank is not None else ''
        if self.config.level == Const.LEVEL_L2:
            create_directory(self.dump_iter_dir)
            kernel_config_path = create_kernel_config_json(self.dump_iter_dir, cur_rank)
            self.config.kernel_config_path = kernel_config_path
            return

        dump_dir = os.path.join(self.dump_iter_dir, f"rank{cur_rank}")
        create_directory(dump_dir)
        if self.config.task in self.data_collector.tasks_need_tensor_data:
            dump_data_dir = os.path.join(dump_dir, "dump_tensor_data")
            create_directory(dump_data_dir)
        else:
            dump_data_dir = None

        dump_file_path = os.path.join(dump_dir, "dump.json")
        stack_file_path = os.path.join(dump_dir, "stack.json")
        construct_file_path = os.path.join(dump_dir, "construct.json")
        free_benchmark_file_path = os.path.join(self.config.dump_path, "free_benchmark.csv")
        self.data_collector.update_dump_paths(
            dump_file_path, stack_file_path, construct_file_path, dump_data_dir, free_benchmark_file_path)

    def register_hook_new(self):
        logger.info_on_rank_0("The {} hook function is successfully mounted to the model.".format(self.config.task))
        if self.config.level in [Const.LEVEL_L0, Const.LEVEL_MIX]:
            self.module_processor.hook_modules(self.model, self.build_hook)

        if self.config.level in [Const.LEVEL_MIX, Const.LEVEL_L1, Const.LEVEL_L2]:
            api_register.initialize_hook(
                functools.partial(self.build_hook, BaseScope.Module_Type_API),
                self.config.online_run_ut
            )
            api_register.api_modularity()

    def attl_init(self):
        if self.config.online_run_ut:
            from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.attl import ATTLConfig, ATTL
            attl_config = ATTLConfig(is_benchmark_device=False,
                                     connect_ip=self.config.host,
                                     connect_port=self.config.port,
                                     nfs_path=self.config.nfs_path,
                                     tls_path=self.config.tls_path)
            need_dump = len(self.config.rank) == 0 or self.current_rank in self.config.rank
            self.attl = ATTL('npu', attl_config, need_dump=need_dump)
            if self.config.nfs_path:
                self.attl.upload("start")

    def attl_send(self, api_data):
        logger.info(f"tools is dumping api: {api_data.name}, rank: {self.current_rank}")
        api_type, _, _ = api_data.name.split(Const.SEP)
        if api_type in [Const.DISTRIBUTED]:
            logger.info(f"api {api_data.name} is not supported, skip")
            return
        if self.config.nfs_path:
            self.attl.upload(api_data)
        else:
            self.attl.send(api_data)

    def attl_stop(self):
        if self.config.nfs_path:
            self.attl.upload("end")
        elif self.attl.socket_manager is not None:
            logger.info(f"pid: {os.getpid()} finished, start send STOP signal.")
            self.attl.socket_manager.send_stop_signal()

    def reset_status(self):
        ModuleProcesser.reset_module_stats()
        HOOKModule.reset_module_stats()
        self.data_collector.data_writer.reset_cache()

        if self.config.level == Const.LEVEL_L2:
            self.data_collector.data_processor.reset_status()
            return
        if self.config.step and self.current_iter not in self.config.step:
            return
        if self.config.rank and self.current_rank not in self.config.rank:
            return
        if self.config.level in [Const.LEVEL_MIX, Const.LEVEL_L0] and self.model:
            for single_model in self.model:
                for _, module in single_model.named_modules():
                    if module == single_model:
                        continue
                    if hasattr(module, 'has_param_hook'):
                        del module.has_param_hook
