# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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
from mindspore.ops.primitive import Primitive

try:
    from mindspore.common._pijit_context import PIJitCaptureContext
except ImportError:
    pijit_label = False
else:
    pijit_label = True

from msprobe.core.common.exceptions import DistributedNotInitializedError, MsprobeException
from msprobe.core.common.file_utils import create_directory
from msprobe.core.common.utils import Const, print_tools_ends_info, DumpPathAggregation
from msprobe.core.data_dump.data_collector import build_data_collector
from msprobe.core.data_dump.data_processor.base import (ModuleBackwardInputsOutputs, ModuleForwardInputsOutputs,
                                                        ModuleBackwardInputs)
from msprobe.core.data_dump.scope import BaseScope
from msprobe.mindspore.cell_processor import CellProcessor
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.common.utils import (get_rank_if_initialized, clean_input_kwargs,
                                            is_mindtorch, register_backward_hook_functions)
from msprobe.mindspore.dump.hook_cell.api_registry import api_register
from msprobe.mindspore.dump.hook_cell.primitive_hooks import PrimitiveHookService
from msprobe.mindspore.dump.jit_dump import JitDump
from msprobe.mindspore.dump.hook_cell.hook_cell import HOOKCell
from msprobe.mindspore.dump.kernel_dump.kernel_config import create_kernel_config_json

if is_mindtorch():
    import torch


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
        self.should_stop_service = False
        self.params_grad_info = {}
        self.hook_handle_dict = {}
        # 提前注册，确保注册尽可能多的API hook
        self.register_api_hook()
        self.init_for_debug_level()

    @staticmethod
    def check_model_valid(models):
        target_module_type = (torch.nn.Module, "torch.nn.Module") if is_mindtorch() else (nn.Cell, "mindspore.nn.Cell")
        if models is None or isinstance(models, target_module_type[0]):
            return models
        error_model = None
        if isinstance(models, (list, tuple)):
            for model in models:
                if not isinstance(model, target_module_type[0]):
                    error_model = model
                    break
        else:
            error_model = models

        if error_model is not None:
            error_info = (f"The 'model' parameter must be a {target_module_type[1]} or list[{target_module_type[1]}] "
                          f"type, currently there is a {type(error_model)} type.")
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, error_info)
        return models

    @staticmethod
    def prepare_module_input_output(target_type, cell, input_data, output):
        if target_type == BaseScope.Module_Type_Module:
            module_input_output = ModuleForwardInputsOutputs(args=input_data, kwargs={}, output=output)
        else:
            module_input_output = ModuleForwardInputsOutputs(args=input_data, kwargs=cell.input_kwargs, output=output)
        return module_input_output

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

        def register_param_hook(ori_name, cell, params_dict):
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
                        handle = param.register_hook(grad_hook(cell, ori_name, param_name))
                        self.hook_handle_dict[name] = handle

        def init_params_grad_info(cell, params_dict):
            '''
            初始化参数梯度信息, 在前向hook结束后, 将参数梯度信息写入cache_data中用于占位
            '''
            if not params_dict:
                return
            if not (Const.FORWARD in self.config.data_mode and Const.BACKWARD not in self.config.data_mode):
                grad_name = cell.params_grad_name if hasattr(cell, 'params_grad_name') else None
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

        def forward_hook(api_or_cell_name, cell, input_data, output):
            if not self.should_execute_hook(target_type, cell, True):
                clean_input_kwargs(cell)
                return None
            with _no_grad():
                self.inner_switch = True
                module_input_output = self.prepare_module_input_output(target_type, cell, input_data, output)
                if target_type == BaseScope.Module_Type_Module:
                    api_or_cell_name = self.cell_processor.set_and_get_reserved_name(cell, api_or_cell_name)
                    params_dict = {}
                    if self.config.task != Const.STRUCTURE:
                        params_dict = {
                            key.split(Const.SEP)[-1]: value
                            for key, value in cell.parameters_dict(recurse=False).items()
                        }
                        setattr(module_input_output, Const.PARAMS, params_dict)
                    # 判断是否需要注册参数hook
                    if params_dict:
                        ori_name = api_or_cell_name.rsplit(Const.SEP, 2)[0]
                        grad_name = ori_name + Const.SEP + Const.PARAMS_GRAD
                        # 首次执行前向hook时，添加params_grad_name属性，并注册参数hook
                        setattr(cell, 'params_grad_name', grad_name)
                        register_param_hook(ori_name, cell, params_dict)
                    self.data_collector.update_api_or_module_name(api_or_cell_name)
                    self.data_collector.forward_data_collect(api_or_cell_name, cell, pid, module_input_output)
                    init_params_grad_info(cell, params_dict)
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

        def pre_backward_hook(api_or_cell_name, cell, grad_input):
            if not self.should_execute_hook(target_type, cell, False):
                return
            self.inner_switch = True
            module_input = ModuleBackwardInputs(grad_input=grad_input)
            self.data_collector.update_api_or_module_name(api_or_cell_name)
            self.data_collector.backward_input_data_collect(api_or_cell_name, cell, pid, module_input)

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
        pre_backward_hook = functools.partial(pre_backward_hook, full_backward_name)

        def wrap_pre_forward_hook(cell, input_data):
            return pre_forward_hook(cell, input_data)

        def wrap_forward_hook(cell, input_data, output_data):
            return forward_hook(cell, input_data, output_data)

        def wrap_backward_hook(cell, grad_input, grad_output):
            return backward_hook(cell, grad_input, grad_output)

        def wrap_pre_backward_hook(cell, grad_input):
            return pre_backward_hook(cell, grad_input)

        return wrap_pre_forward_hook, wrap_forward_hook, wrap_backward_hook, wrap_pre_backward_hook

    def update_primitive_counters(self, primitive_name):
        if primitive_name not in self.primitive_counters:
            self.primitive_counters[primitive_name] = 0
        else:
            self.primitive_counters[primitive_name] += 1

    def step(self):
        if self.config.level == Const.LEVEL_DEBUG:
            return
        if self.config.async_dump:
            self.data_collector.fill_stack_tensor_data()
            if self.config.task == Const.TENSOR:
                self.data_collector.data_processor.dump_async_data()
        self.data_collector.write_json()
        self.current_iter += 1
        self.data_collector.update_iter(self.current_iter)
        self.reset_status()

    def start(self, model=None):
        if self.config.level == Const.LEVEL_DEBUG:
            return
        self.start_call = True
        if self.should_stop_service:
            return
        if self.need_end_service():
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
            self.register_primitive_hook()
            self.register_cell_hook()
            if self.config.level in [Const.LEVEL_MIX, Const.LEVEL_L1]:
                JitDump.set_config(self.config)
                JitDump.set_data_collector(self.data_collector)
                if hasattr(ms.common.api, "_MindsporeFunctionExecutor"):
                    ms.common.api._MindsporeFunctionExecutor = JitDump
                else:
                    ms.common.api._JitExecutor = JitDump
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
        if self.config.level == Const.LEVEL_DEBUG:
            return
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
        if self.config.async_dump:
            self.data_collector.fill_stack_tensor_data()
            if self.config.task == Const.TENSOR:
                self.data_collector.data_processor.dump_async_data()
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
        self.data_collector.update_dump_paths(dump_path_aggregation)

        self.data_collector.initialize_json_file(
            framework=Const.MT_FRAMEWORK if is_mindtorch() else Const.MS_FRAMEWORK
        )

    def empty(self, *args, **kwargs):
        pass

    def register_api_hook(self):
        if self.config.level in [Const.LEVEL_MIX, Const.LEVEL_L1, Const.LEVEL_L2]:
            logger.info(f"The api {self.config.task} hook function is successfully mounted to the model.")
            api_register.initialize_hook(functools.partial(self.build_hook, BaseScope.Module_Type_API))
            api_register.api_set_hook_func()

    def get_cells_and_names(self):
        cells_and_names_with_index = {}

        def get_cell_or_module(model):
            return model.named_modules() if is_mindtorch() else model.cells_and_names()

        if isinstance(self.model, (list, tuple)):
            for index, model in enumerate(self.model):
                cells_and_names_with_index[str(index)] = get_cell_or_module(model)
        else:
            cells_and_names_with_index["-1"] = get_cell_or_module(self.model)
        return cells_and_names_with_index

    def register_primitive_hook(self):
        if self.config.level not in [Const.LEVEL_MIX, Const.LEVEL_L1]:
            return
        if not self.model or self.config.task not in Const.DUMP_DATA_COLLECTION_LIST:
            return

        primitive_set = set()
        cells_and_names_with_index = self.get_cells_and_names()
        for cells_and_names in cells_and_names_with_index.values():
            for _, cell in cells_and_names:
                for attribute, value in vars(cell).items():
                    if isinstance(value, Primitive):
                        primitive_set.add((attribute, value))

        for pname, primitive in primitive_set:
            primitive_class_name = primitive.__class__.__name__
            primitive_combined_name = pname + Const.SEP + primitive_class_name
            new_primitive = type('NewPrimitive', (primitive.__class__,),
                                 {'__call__': self.primitive_hook_service.wrap_primitive(primitive.__call__,
                                                                                         primitive_combined_name)})
            primitive.__class__ = new_primitive

    def register_cell_hook(self):
        if self.config.level in [Const.LEVEL_MIX, Const.LEVEL_L0]:
            logger.info(f"The cell {self.config.task} hook function is successfully mounted to the model.")
            if not self.model:
                raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                       f"The current level is {self.config.level}, the model cannot be None")
            model_type = Const.MODULE if is_mindtorch() else Const.CELL
            cells_and_names_with_index = self.get_cells_and_names()

            for index, cells_and_names in cells_and_names_with_index.items():
                model = self.model if index == "-1" else self.model[int(index)]
                for name, cell in cells_and_names:
                    if cell == model:
                        continue
                    cell_index = (index + Const.SEP) if index != "-1" else ""
                    prefix = (model_type + Const.SEP + cell_index + name +
                              Const.SEP + cell.__class__.__name__ + Const.SEP)
                    _, forward_hook, backward_hook, _ = self.build_hook(BaseScope.Module_Type_Module, prefix)
                    cell.register_forward_hook(forward_hook)
                    cell.register_forward_pre_hook(
                        self.cell_processor.node_hook(prefix + Const.FORWARD, Const.START))
                    cell.register_forward_hook(
                        self.cell_processor.node_hook(prefix + Const.FORWARD, Const.STOP))

                    register_backward_hook_functions["full"](cell, backward_hook)
                    register_backward_hook_functions["pre"](
                        cell, self.cell_processor.node_hook(prefix + Const.BACKWARD, Const.START))
                    register_backward_hook_functions["full"](
                        cell, self.cell_processor.node_hook(prefix + Const.BACKWARD, Const.STOP))

    def reset_status(self):
        self.primitive_hook_service.primitive_counters.clear()
        self.data_collector.reset_status()
        JitDump.jit_count = defaultdict(int)
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
        self.data_collector.initialize_json_file(
            framework=Const.MT_FRAMEWORK if is_mindtorch() else Const.MS_FRAMEWORK
        )
        self.debug_variable_counter = defaultdict(int)

    def save(self, variable, name, save_backward):
        '''
        Args:
            variable: Union[List[variable], dict{str: variable}, mindspore.tensor, str, float, int]
            name: str
            save_backward: boolean
        Return:
            void
        '''
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
