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
from msprobe.core.common.exceptions import DistributedNotInitializedError, MsprobeException
from msprobe.core.common.file_utils import create_directory
from msprobe.core.common.utils import print_tools_ends_info
from msprobe.core.data_dump.data_collector import build_data_collector
from msprobe.core.data_dump.data_processor.base import ModuleForwardInputsOutputs, ModuleBackwardInputsOutputs
from msprobe.core.data_dump.scope import BaseScope
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import get_rank_if_initialized
from msprobe.pytorch.hook_module import remove_dropout
from msprobe.pytorch.hook_module.api_registry import api_register
from msprobe.pytorch.hook_module.hook_module import HOOKModule
from msprobe.pytorch.module_processer import ModuleProcesser
from msprobe.pytorch.api_accuracy_checker.common.utils import ApiData

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
        self.current_iter = 0
        self.first_start = True
        self.current_rank = None
        self.dump_iter_dir = None
        self.should_stop_service = False
        self.attl = None

    @staticmethod
    def forward_backward_dump_end():
        logger.info_on_rank_0("Data needed ends here.")
        api_register.api_originality()

    @staticmethod
    def is_registered_backward_hook(module):
        if hasattr(module, '_backward_hooks') and \
                len(module._backward_hooks) > 0 and \
                module._is_full_backward_hook is False:
            return True
        return False

    def check_register_full_backward_hook(self, module):
        if self.is_registered_backward_hook(module):
            module._backward_hooks.clear()
            module._is_full_backward_hook = None
            logger.warning("Found deprecated backward hooks. Removing them and switching to full backward hooks.")

    def build_hook(self, module_type, name):
        def pre_hook(api_or_module_name, module, args, kwargs):
            if not self.should_execute_hook():
                return args, kwargs

            if module_type == BaseScope.Module_Type_Module:
                api_or_module_name = module.mindstudio_reserved_name
            self.data_collector.update_api_or_module_name(api_or_module_name)

            if self.config.online_run_ut:
                return None, None
            if self.data_collector:
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=None)
                self.data_collector.pre_forward_data_collect(api_or_module_name, module, pid, module_input_output)
            return args, kwargs

        def forward_hook(api_or_module_name, module, args, kwargs, output):
            if not self.should_execute_hook():
                return None

            if module_type == BaseScope.Module_Type_Module:
                api_or_module_name = module.mindstudio_reserved_name
            self.data_collector.update_api_or_module_name(api_or_module_name)

            if self.config.online_run_ut:
                if self.data_collector.scope and not self.data_collector.scope.check(api_or_module_name):
                    return None
                api_data = ApiData(name[:-1], args, kwargs, output, self.current_iter, self.current_rank)
                self.attl_send(api_data)
                return None

            if self.data_collector:
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=output)
                self.data_collector.forward_data_collect(api_or_module_name, module, pid, module_input_output)
                if self.data_collector.if_return_forward_new_output():
                    return self.data_collector.get_forward_new_output()
            return output

        def forward_hook_torch_version_below_2(api_or_module_name, module, args, output):
            return forward_hook(api_or_module_name, module, args, {}, output)

        def backward_hook(api_or_module_name, module, grad_input, grad_output):
            if not self.should_execute_hook():
                return

            if module_type == BaseScope.Module_Type_Module:
                api_or_module_name = module.mindstudio_reserved_name
            self.data_collector.update_api_or_module_name(api_or_module_name)

            if self.config.online_run_ut:
                return

            if self.data_collector:
                # 此处获取到的grad_input实际为反向过程的输出数据，grad_output为反向过程的输入数据，因此传入时调换顺序
                module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_output, grad_output=grad_input)
                self.data_collector.backward_data_collect(api_or_module_name, module, pid, module_input_output)

        pid = os.getpid()
        forward_name_template = name + Const.FORWARD
        backward_name_template = name + Const.BACKWARD
        pre_forward_hook_fn = functools.partial(pre_hook, forward_name_template)
        forward_hook_fn = functools.partial(forward_hook, forward_name_template)
        backward_hook_fn = functools.partial(backward_hook, backward_name_template)
        forward_hook_torch_version_below_2_fn = functools.partial(forward_hook_torch_version_below_2,
                                                                  forward_name_template)
        return HookFn(pre_forward_hook_fn, forward_hook_fn, backward_hook_fn, forward_hook_torch_version_below_2_fn)

    def start(self, model, api_origin=False):
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
        if api_origin:
            api_register.api_modularity()
        if self.config.online_run_ut and torch_version_above_or_equal_2:
            run_ut_dispatch(self.attl, True)
        self.switch = True
        logger.info_on_rank_0(f"Dump switch is turned on at step {self.current_iter}. ")
        if self.config.level != "L2" and not self.config.online_run_ut:
            self.create_dirs()
            logger.info_on_rank_0(f"Dump data will be saved in {self.dump_iter_dir}.")

    def stop(self):
        if self.should_stop_service:
            return
        if self.config.level == "L2":
            return
        if self.config.step and self.current_iter not in self.config.step:
            return
        if self.config.rank and self.current_rank not in self.config.rank:
            return
        self.switch = False
        if self.config.online_run_ut and torch_version_above_or_equal_2:
            run_ut_dispatch(self.attl, False)
            return
        self.data_collector.write_json()

    def step(self):
        if self.should_stop_service:
            return
        self.current_iter += 1
        self.data_collector.update_iter(self.current_iter)

        ModuleProcesser.reset_module_stats()
        HOOKModule.reset_module_stats()
        self.data_collector.data_writer.reset_cache()

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

    def should_execute_hook(self):
        if not self.switch:
            return False
        if self.data_collector and self.data_collector.data_processor.is_terminated:
            return False
        return True

    def create_dirs(self):
        create_directory(self.config.dump_path)
        self.dump_iter_dir = os.path.join(self.config.dump_path, f"step{self.current_iter}")
        cur_rank = self.current_rank if self.current_rank is not None else ''
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
        if self.config.level in ["L0", "mix"]:
            if self.model is None:
                logger.error_log_with_exp("The model is None.", MsprobeException.INVALID_PARAM_ERROR)
            logger.info_on_rank_0("The init dump mode is enabled, and the module dump function will not be available")
            for name, module in self.model.named_modules():
                if module == self.model:
                    continue
                prefix = BaseScope.Module_Type_Module + Const.SEP + name + Const.SEP + \
                         module.__class__.__name__ + Const.SEP

                pre_forward_hook, forward_hook, backward_hook, forward_hook_torch_version_below_2 = self.build_hook(
                    BaseScope.Module_Type_Module, prefix)
                if torch_version_above_or_equal_2:
                    module.register_forward_hook(forward_hook, with_kwargs=True)
                else:
                    self.check_register_full_backward_hook(module)
                    module.register_full_backward_hook(
                        self.module_processor.node_hook(prefix + Const.BACKWARD, Const.STOP))
                    module.register_forward_hook(forward_hook_torch_version_below_2)
                self.check_register_full_backward_hook(module)
                module.register_full_backward_hook(backward_hook)

                module.register_forward_pre_hook(
                    self.module_processor.node_hook(prefix + Const.FORWARD, Const.START))
                module.register_forward_hook(
                    self.module_processor.node_hook(prefix + Const.FORWARD, Const.STOP))
                if torch_version_above_or_equal_2:
                    module.register_full_backward_pre_hook(
                        self.module_processor.node_hook(prefix + Const.BACKWARD, Const.START))
                    self.check_register_full_backward_hook(module)
                    module.register_full_backward_hook(
                        self.module_processor.node_hook(prefix + Const.BACKWARD, Const.STOP))

        if self.config.level in ["mix", "L1", "L2"]:
            api_register.initialize_hook(functools.partial(self.build_hook, BaseScope.Module_Type_API),
                                         self.config.online_run_ut)
            api_register.api_modularity()

        if Const.STATISTICS == self.config.task or Const.TENSOR == self.config.task:
            remove_dropout()

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
