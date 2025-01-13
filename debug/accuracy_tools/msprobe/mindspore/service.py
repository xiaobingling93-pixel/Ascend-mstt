# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import copy
import functools
import os
from collections import defaultdict

import mindspore as ms
from mindspore import nn
from mindspore.common.api import _no_grad
try:
    from mindspore.common._pijit_context import PIJitCaptureContext
except ImportError:
    pijit_label = False
else:
    pijit_label = True

from msprobe.core.common.exceptions import DistributedNotInitializedError, MsprobeException
from msprobe.core.common.file_utils import create_directory
from msprobe.core.common.utils import Const, print_tools_ends_info
from msprobe.core.data_dump.data_collector import build_data_collector
from msprobe.core.data_dump.data_processor.base import ModuleBackwardInputsOutputs, ModuleForwardInputsOutputs
from msprobe.core.data_dump.scope import BaseScope
from msprobe.mindspore.cell_processor import CellProcessor
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.common.utils import get_rank_if_initialized, clean_input_kwargs
from msprobe.mindspore.dump.hook_cell.api_registry import api_register
from msprobe.mindspore.dump.hook_cell.primitive_hooks import PrimitiveHookService
from msprobe.mindspore.dump.jit_dump import JitDump
from msprobe.mindspore.dump.hook_cell.hook_cell import HOOKCell


class Service:
    def __init__(self, config):
        self.model = None
        self.config = copy.deepcopy(config)
        self.config.level = self.config.level_ori
        self.data_collector = build_data_collector(self.config)
        self.cell_processor = CellProcessor(self.data_collector.scope)
        self.primitive_hook_service = PrimitiveHookService(self)
        self.switch = False
        self.inner_switch = False
        self.primitive_switch = False
        self.current_iter = 0
        self.first_start = True
        self.current_rank = None
        self.dump_iter_dir = None
        self.start_call = False
        self.check_level_valid()
        self.should_stop_service = False

    @staticmethod
    def check_model_valid(model):
        if not model or isinstance(model, nn.Cell):
            return model
        raise MsprobeException(
            MsprobeException.INVALID_PARAM_ERROR, "model 参数必须是 mindspore.nn.Cell 类型。"
        )

    @staticmethod
    def prepare_module_input_output(target_type, cell, input_data, output):
        if target_type == BaseScope.Module_Type_Module:
            module_input_output = ModuleForwardInputsOutputs(args=input_data, kwargs={}, output=output)
        else:
            module_input_output = ModuleForwardInputsOutputs(args=input_data, kwargs=cell.input_kwargs, output=output)
        return module_input_output

    def check_level_valid(self):
        if self.config.level == Const.LEVEL_L2:
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, "L2 level dump function is currently not supported."
            )

    def build_hook(self, target_type, name):
        def pre_hook(api_or_cell_name, cell, input_data):
            if not self.should_execute_hook(target_type, cell, True):
                clean_input_kwargs(cell)
                return None

            with _no_grad():
                self.inner_switch = True
                if target_type == BaseScope.Module_Type_Module:
                    api_or_cell_name = self.cell_processor.set_and_get_reserved_name(cell, api_or_cell_name)
                else:
                    cell.forward_data_collected = True
                    HOOKCell.add_cell_count(name)
                module_input_output = self.prepare_module_input_output(target_type, cell, input_data, None)
                self.data_collector.update_api_or_module_name(api_or_cell_name)
                self.data_collector.forward_input_data_collect(api_or_cell_name, cell, pid, module_input_output)
                self.inner_switch = False
                return input_data

        def grad_hook(cell, ori_name, param_name):
            def hook_fn(grad):
                if not self.should_execute_hook(target_type, cell, False):
                    return None
                self.inner_switch = True
                self.data_collector.params_data_collect(ori_name, param_name, pid, grad)
                self.inner_switch = False
                return None

            return hook_fn

        def register_param_hook(cell_name, cell, params_dict):
            # data_mode为forward时，不注册参数hook
            if not (Const.FORWARD in self.config.data_mode and Const.BACKWARD not in self.config.data_mode):
                # 判断参数是否已经注册过hook
                if params_dict and hasattr(cell, 'has_param_hook') and not cell.has_param_hook:
                    ori_name = cell_name.rsplit(Const.SEP, 2)[0]
                    grad_name = ori_name + Const.SEP + Const.PARAMS_GRAD
                    # 注册hook时，初始化grad_name的data_info
                    data_info = {grad_name: {key: [None] for key in params_dict}}
                    # 将grad_name的data_info先写入cache_data中, 梯度计算后再更新
                    self.data_collector.handle_data(grad_name, data_info,
                                                    flush=self.data_collector.data_processor.is_terminated)
                    for param_name, param in params_dict.items():
                        param.register_hook(grad_hook(cell, ori_name, param_name))
                    cell.has_param_hook = True

        def forward_hook(api_or_cell_name, cell, input_data, output):
            if not self.should_execute_hook(target_type, cell, True):
                clean_input_kwargs(cell)
                return None
            with _no_grad():
                self.inner_switch = True
                module_input_output = self.prepare_module_input_output(target_type, cell, input_data, output)
                if target_type == BaseScope.Module_Type_Module:
                    api_or_cell_name = self.cell_processor.set_and_get_reserved_name(cell, api_or_cell_name)
                    params_dict = {key.split(Const.SEP)[-1]: value for key, value in cell.parameters_dict(recurse=False).items()}
                    setattr(module_input_output, Const.PARAMS, params_dict)
                    # 设置has_param_hook属性，避免重复注册hook
                    if not hasattr(cell, 'has_param_hook'):
                        setattr(cell, 'has_param_hook', False)
                    self.data_collector.update_api_or_module_name(api_or_cell_name)
                    self.data_collector.forward_data_collect(api_or_cell_name, cell, pid, module_input_output)
                    register_param_hook(api_or_cell_name, cell, params_dict)
                else:
                    self.data_collector.update_api_or_module_name(api_or_cell_name)
                    self.data_collector.forward_output_data_collect(api_or_cell_name, cell, pid, module_input_output)

                if self.data_collector.if_return_forward_new_output():
                    forward_new_output = self.data_collector.get_forward_new_output()
                    self.inner_switch = False
                    return forward_new_output
                clean_input_kwargs(cell)
                self.inner_switch = False
                return output

        def backward_hook(api_or_cell_name, cell, grad_input, grad_output):
            if not self.should_execute_hook(target_type, cell, False):
                return
            self.inner_switch = True

            need_exchange = True
            if target_type == BaseScope.Module_Type_Module:
                if not hasattr(cell, 'has_pre_hook_called') or not cell.has_pre_hook_called:
                    need_exchange = False
                api_or_cell_name = self.cell_processor.set_and_get_reserved_name(cell, api_or_cell_name)

            self.data_collector.update_api_or_module_name(api_or_cell_name)
            if self.data_collector:
                # 框架最新接口变更，grad_input和grad_output的含义发生了变化，与torch含义保持一致，因此此处调换顺序传入
                if need_exchange:
                    module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_output, grad_output=grad_input)
                else:
                    module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_input, grad_output=grad_output)
                self.data_collector.backward_data_collect(api_or_cell_name, cell, pid, module_input_output)
            self.inner_switch = False

        pid = os.getpid()
        if target_type == BaseScope.Module_Type_Module:
            full_forward_name = name + Const.FORWARD
            full_backward_name = name + Const.BACKWARD
        else:
            full_forward_name = name + str(HOOKCell.get_cell_count(name)) + Const.SEP + Const.FORWARD
            full_backward_name = name + str(HOOKCell.get_cell_count(name)) + Const.SEP + Const.BACKWARD
        pre_forward_hook = functools.partial(pre_hook, full_forward_name)
        forward_hook = functools.partial(forward_hook, full_forward_name)
        backward_hook = functools.partial(backward_hook, full_backward_name)

        def wrap_pre_forward_hook(cell, input_data):
            return pre_forward_hook(cell, input_data)

        def wrap_forward_hook(cell, input_data, output_data):
            return forward_hook(cell, input_data, output_data)

        def wrap_backward_hook(cell, grad_input, grad_output):
            return backward_hook(cell, grad_input, grad_output)

        return wrap_pre_forward_hook, wrap_forward_hook, wrap_backward_hook

    def update_primitive_counters(self, primitive_name):
        if primitive_name not in self.primitive_counters:
            self.primitive_counters[primitive_name] = 0
        else:
            self.primitive_counters[primitive_name] += 1

    def register_primitive_hooks(self):
        primitive_set = set()
        for _, cell in self.model.cells_and_names():
            for pname, primitive in cell._primitives.items():
                primitive_set.add((pname, primitive))

        for pname, primitive in primitive_set:
            primitive_class_name = primitive.__class__.__name__
            primitive_combined_name = pname + Const.SEP + primitive_class_name
            new_primitive = type('NewPrimitive', (primitive.__class__,),
                                 {'__call__': self.primitive_hook_service.wrap_primitive(primitive.__call__,
                                                                                         primitive_combined_name)})
            primitive.__class__ = new_primitive

    def step(self):
        self.data_collector.write_json()
        self.current_iter += 1
        self.data_collector.update_iter(self.current_iter)
        self.reset_status()

    def start(self, model=None):
        self.start_call = True
        if self.should_stop_service:
            return
        if self.need_end_service():
            api_register.api_set_ori_func()
            self.should_stop_service = True
            self.switch = False
            self.primitive_switch = False
            print_tools_ends_info()
            return
        if self.config.step and self.current_iter not in self.config.step:
            return
        self.model = self.check_model_valid(model)

        logger.info(f"{Const.TOOL_NAME}: debugger.start() is set successfully")

        if self.first_start:
            try:
                self.current_rank = get_rank_if_initialized()
            except DistributedNotInitializedError:
                self.current_rank = None

            if self.config.rank and self.current_rank not in self.config.rank:
                return
            self.register_hook_new()
            if self.config.level in [Const.LEVEL_MIX, Const.LEVEL_L1]:
                JitDump.set_config(self.config)
                JitDump.set_data_collector(self.data_collector)
                ms.common.api._MindsporeFunctionExecutor = JitDump
                ms.common.api._PyNativeExecutor.grad = JitDump.grad
                if pijit_label:
                    PIJitCaptureContext.__enter__ = self.empty
                    PIJitCaptureContext.__exit__ = self.empty
            self.first_start = False

        api_register.api_set_hook_func()
        self.switch = True
        self.primitive_switch = True
        logger.info(f"Dump switch is turned on at step {self.current_iter}. ")
        self.create_dirs()
        logger.info(f"Dump data will be saved in {self.dump_iter_dir}.")
        JitDump.jit_dump_switch = True

    def stop(self):
        if self.should_stop_service:
            return
        logger.info(f"{Const.TOOL_NAME}: debugger.stop() is set successfully. "
                    "Please set debugger.start() to turn on the dump switch again. ")
        if not self.start_call:
            logger.error(f"{Const.TOOL_NAME}: debugger.start() is not set in the current scope.")
            raise Exception("debugger.start() is not set in the current scope.")
        if self.config.step and self.current_iter not in self.config.step:
            return
        if self.config.rank and self.current_rank not in self.config.rank:
            return
        self.switch = False
        self.primitive_switch = False
        self.start_call = False
        self.data_collector.write_json()
        JitDump.jit_dump_switch = False

    def need_end_service(self):
        if self.config.step and self.current_iter > max(self.config.step):
            return True
        if self.data_collector and self.data_collector.data_processor.is_terminated:
            return True
        return False

    def should_execute_hook(self, hook_type, cell, is_forward):
        is_cell_hook = hook_type == BaseScope.Module_Type_Module
        if is_cell_hook and not self.switch:
            return False
        elif not is_cell_hook and is_forward and not self.switch:
            return False
        elif not is_cell_hook and not is_forward and not cell.forward_data_collected:
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
        self.data_collector.update_dump_paths(
            dump_file_path, stack_file_path, construct_file_path, dump_data_dir, None)

    def empty(self, *args, **kwargs):
        pass

    def register_hook_new(self):
        logger.info("The {} hook function is successfully mounted to the model.".format(self.config.task))
        if self.config.level in [Const.LEVEL_MIX, Const.LEVEL_L1]:
            api_register.initialize_hook(functools.partial(self.build_hook, BaseScope.Module_Type_API))
            api_register.api_set_hook_func()
            if self.model and self.config.task in Const.DUMP_DATA_COLLECTION_LIST:
                self.register_primitive_hooks()

        if self.config.level in [Const.LEVEL_MIX, Const.LEVEL_L0]:
            if not self.model:
                raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                       f"The current level is {self.config.level}, the model cannot be None")
            for name, cell in self.model.cells_and_names():
                if cell == self.model:
                    continue

                prefix = 'Cell' + Const.SEP + name + Const.SEP + \
                         cell.__class__.__name__ + Const.SEP
                _, forward_hook, backward_hook = self.build_hook(BaseScope.Module_Type_Module, prefix)
                cell.register_forward_hook(forward_hook)
                cell.register_backward_hook(backward_hook)

                cell.register_forward_pre_hook(
                    self.cell_processor.node_hook(prefix + Const.FORWARD, Const.START))
                cell.register_forward_hook(
                    self.cell_processor.node_hook(prefix + Const.FORWARD, Const.STOP))
                cell.register_backward_pre_hook(
                    self.cell_processor.node_hook(prefix + Const.BACKWARD, Const.START))
                cell.register_backward_hook(
                    self.cell_processor.node_hook(prefix + Const.BACKWARD, Const.STOP))

    def reset_status(self):
        self.primitive_hook_service.primitive_counters.clear()
        self.data_collector.data_writer.reset_cache()
        JitDump.jit_count = defaultdict(int)

        if self.config.step and self.current_iter not in self.config.step:
            return
        if self.config.rank and self.current_rank not in self.config.rank:
            return

        if self.config.level in [Const.LEVEL_MIX, Const.LEVEL_L0] and self.model:
            for _, cell in self.model.cells_and_names():
                if cell == self.model:
                    continue
                if hasattr(cell, 'has_param_hook'):
                    del cell.has_param_hook
