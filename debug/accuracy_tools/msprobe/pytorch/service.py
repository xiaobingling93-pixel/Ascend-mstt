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

import functools
import os
from collections import namedtuple, defaultdict

import torch
from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import DistributedNotInitializedError
from msprobe.core.common.file_utils import create_directory
from msprobe.core.common.utils import print_tools_ends_info, DumpPathAggregation
from msprobe.core.data_dump.data_collector import build_data_collector
from msprobe.core.data_dump.data_processor.base import ModuleForwardInputsOutputs, ModuleBackwardInputsOutputs
from msprobe.core.data_dump.scope import BaseScope
from msprobe.pytorch.api_accuracy_checker.common.utils import ApiData
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import get_rank_if_initialized, is_recomputation
from msprobe.pytorch.dump.kernel_dump.kernel_config import create_kernel_config_json
from msprobe.pytorch.dump.module_dump.module_processer import ModuleProcesser
from msprobe.pytorch.hook_module.api_registry import api_register
from msprobe.pytorch.hook_module.hook_module import HOOKModule
from msprobe.pytorch.hook_module.register_optimizer_hook import register_optimizer_hook

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
        self.params_grad_info = {}
        self.hook_handle_dict = {}
        # 提前注册，确保注册尽可能多的API hook
        self.register_api_hook()
        self.init_for_debug_level()

    def build_hook(self, module_type, name):
        def pre_hook(api_or_module_name, module, args, kwargs):
            if not self.should_execute_hook(module_type, module, True):
                return args, kwargs
            is_recompute = is_recomputation()

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
                self.data_collector.forward_input_data_collect(
                    api_or_module_name,
                    module,
                    pid,
                    module_input_output,
                    is_recompute
                )

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

        def register_param_hook(ori_name, module, params_dict):
            '''
            注册参数hook
            '''
            # data_mode为forward时，不注册参数hook
            if not (Const.FORWARD in self.config.data_mode and Const.BACKWARD not in self.config.data_mode):
                for param_name, param in params_dict.items():
                    if param.requires_grad:
                        name = ori_name + Const.SEP + param_name
                        old_handle = self.hook_handle_dict.get(name)
                        if old_handle and hasattr(old_handle, "remove"):
                            old_handle.remove()
                        handle = param.register_hook(grad_hook(module, ori_name, param_name))
                        self.hook_handle_dict[name] = handle

        def init_params_grad_info(module, params_dict):
            '''
            初始化参数梯度信息, 在前向hook结束后, 将参数梯度信息写入cache_data中用于占位
            '''
            if not params_dict:
                return
            if not (Const.FORWARD in self.config.data_mode and Const.BACKWARD not in self.config.data_mode):
                grad_name = module.params_grad_name if hasattr(module, 'params_grad_name') else None
                # 判断是否已经在cache_data中进行了占位, 若没有则先写入cache_data中
                if not self.params_grad_info.get(grad_name):
                    data_info = {grad_name: {key: [None] for key, value in params_dict.items() if value.requires_grad}}
                    # 当模块中的参数有requires_grad属性为True时，才会进行梯度计算，此时才需要占位
                    if data_info.get(grad_name):
                        # 将grad_name的data_info先写入cache_data中, 梯度计算后再更新
                        self.data_collector.handle_data(grad_name, data_info,
                                                        flush=self.data_collector.data_processor.is_terminated)
                    # 记录当前模块的参数梯度信息已占位
                    self.params_grad_info[grad_name] = True

        def forward_hook(api_or_module_name, module, args, kwargs, output):
            if not self.should_execute_hook(module_type, module, True):
                return None
            is_recompute = is_recomputation()

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
                params_dict = {}
                if self.config.task != Const.STRUCTURE:
                    params_dict = {
                        key.split(Const.SEP)[-1]: value
                        for key, value in module.named_parameters(recurse=False)
                    }
                    setattr(module_input_output, Const.PARAMS, params_dict)
                # 判断是否需要注册参数hook
                if params_dict:
                    ori_name = api_or_module_name.rsplit(Const.SEP, 2)[0]
                    grad_name = ori_name + Const.SEP + Const.PARAMS_GRAD
                    # 首次执行前向hook时，添加params_grad_name属性，并注册参数hook
                    setattr(module, 'params_grad_name', grad_name)
                    register_param_hook(ori_name, module, params_dict)
                self.data_collector.forward_data_collect(
                    api_or_module_name,
                    module,
                    pid,
                    module_input_output,
                    is_recompute
                )
                init_params_grad_info(module, params_dict)
            else:
                self.data_collector.update_api_or_module_name(api_or_module_name)
                self.data_collector.forward_output_data_collect(
                    api_or_module_name,
                    module,
                    pid,
                    module_input_output,
                    is_recompute
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
            is_recompute = is_recomputation()

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
                self.data_collector.backward_data_collect(
                    api_or_module_name,
                    module,
                    pid,
                    module_input_output,
                    is_recompute
                )
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
        if self.config.level == Const.LEVEL_DEBUG:
            return
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
            self.register_module_hook()
            if self.config.level == Const.LEVEL_MIX:
                register_optimizer_hook(self.data_collector)
            self.first_start = False
        if self.config.online_run_ut and torch_version_above_or_equal_2:
            run_ut_dispatch(self.attl, True, self.config.online_run_ut_recompute)
        self.switch = True
        logger.info_on_rank_0(f"Dump switch is turned on at step {self.current_iter}. ")
        if not self.config.online_run_ut:
            self.create_dirs()
            logger.info_on_rank_0(f"Dump data will be saved in {self.dump_iter_dir}.")

    def stop(self):
        if self.config.level == Const.LEVEL_DEBUG:
            return
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
        if self.config.async_dump:
            self.data_collector.fill_stack_tensor_data()
            if self.config.task == Const.TENSOR:
                self.data_collector.data_processor.dump_async_data()
        self.data_collector.write_json()

    def step(self):
        if self.config.level == Const.LEVEL_DEBUG:
            return
        if self.should_stop_service:
            return
        if self.config.async_dump:
            self.data_collector.fill_stack_tensor_data()
            if self.config.task == Const.TENSOR:
                self.data_collector.data_processor.dump_async_data()
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

        dump_path_aggregation = DumpPathAggregation()
        dump_path_aggregation.dump_file_path = os.path.join(dump_dir, "dump.json")
        dump_path_aggregation.stack_file_path = os.path.join(dump_dir, "stack.json")
        dump_path_aggregation.construct_file_path = os.path.join(dump_dir, "construct.json")
        dump_path_aggregation.dump_tensor_data_dir = dump_data_dir
        dump_path_aggregation.free_benchmark_file_path = os.path.join(dump_dir, "free_benchmark.csv")
        self.data_collector.update_dump_paths(dump_path_aggregation)
        self.data_collector.initialize_json_file(framework=Const.PT_FRAMEWORK)

    def register_api_hook(self):
        if self.config.level in [Const.LEVEL_MIX, Const.LEVEL_L1, Const.LEVEL_L2]:
            logger.info_on_rank_0(f"The api {self.config.task} hook function is successfully mounted to the model.")
            api_register.initialize_hook(
                functools.partial(self.build_hook, BaseScope.Module_Type_API),
                self.config.online_run_ut
            )
            api_register.api_modularity()

    def register_module_hook(self):
        if self.config.level in [Const.LEVEL_L0, Const.LEVEL_MIX]:
            logger.info_on_rank_0(f"The module {self.config.task} hook function is successfully mounted to the model.")
            self.module_processor.register_module_hook(self.model, self.build_hook)

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
        self.data_collector.reset_status()
        self.params_grad_info.clear()

        if self.config.level == Const.LEVEL_L2:
            self.data_collector.data_processor.reset_status()
            return
        if self.config.step and self.current_iter not in self.config.step:
            return
        if self.config.rank and self.current_rank not in self.config.rank:
            return

    def init_for_debug_level(self):
        if not (self.config.level == Const.LEVEL_DEBUG and self.config.task in [Const.TENSOR, Const.STATISTICS]):
            return
        try:
            self.current_rank = get_rank_if_initialized()
        except DistributedNotInitializedError:
            self.current_rank = None

        # dir: dump_path -- rank{} -- debug.json
        self.dump_iter_dir = self.config.dump_path
        cur_rank = self.current_rank if self.current_rank is not None else ''
        dump_dir = os.path.join(self.dump_iter_dir, f"rank{cur_rank}")
        create_directory(dump_dir)
        if self.config.task in self.data_collector.tasks_need_tensor_data:
            dump_data_dir = os.path.join(dump_dir, "dump_tensor_data")
            create_directory(dump_data_dir)
        else:
            dump_data_dir = None

        dump_path_aggregation = DumpPathAggregation()
        dump_path_aggregation.dump_tensor_data_dir = dump_data_dir
        dump_path_aggregation.debug_file_path = os.path.join(dump_dir, "debug.json")
        self.data_collector.update_dump_paths(dump_path_aggregation)
        self.data_collector.initialize_json_file(framework=Const.PT_FRAMEWORK)

        self.debug_variable_counter = defaultdict(int)

    def save(self, variable, name, save_backward):
        if self.config.level != Const.LEVEL_DEBUG:
            return
        count = self.debug_variable_counter[name]
        self.debug_variable_counter[name] += 1

        name_with_count = f"{name}.{count}"
        grad_name_with_count = f"{name}_grad.{count}"

        # forward save
        self.data_collector.debug_data_collect_forward(variable, name_with_count)

        # backward save
        if save_backward:
            self.data_collector.debug_data_collect_backward(variable, grad_name_with_count)
