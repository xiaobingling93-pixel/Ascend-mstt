# Copyright 2024 Huawei Technologies Co., Ltd
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
# ============================================================================

import os
import copy
import functools
from collections import defaultdict

import mindspore as ms
from mindspore.common.tensor import Tensor
from mindspore import ops
from mindspore import nn
try:
    from mindspore.common._pijit_context import PIJitCaptureContext
    pijit_label = True
except ImportError:
    pijit_label = False


from msprobe.core.data_dump.data_collector import build_data_collector
from msprobe.core.data_dump.scope import BaseScope
from msprobe.mindspore.common.utils import get_rank_if_initialized
from msprobe.core.common.file_utils import create_directory
from msprobe.mindspore.common.log import logger
from msprobe.core.common.utils import Const, print_tools_ends_info
from msprobe.core.common.exceptions import DistributedNotInitializedError
from msprobe.mindspore.dump.hook_cell.api_registry import api_register
from msprobe.mindspore.dump.hook_cell.primitive_hooks import PrimitiveHookService
from msprobe.core.data_dump.data_processor.base import ModuleBackwardInputsOutputs, ModuleForwardInputsOutputs, \
    ModuleBackwardInputs, ModuleBackwardOutputs
from msprobe.core.common.exceptions import MsprobeException
from msprobe.mindspore.dump.hook_cell.hook_cell import HOOKCell
from msprobe.mindspore.cell_processor import CellProcessor
from msprobe.mindspore.dump.jit_dump import JitDump


class Service:
    def __init__(self, config):
        self.model = None
        self.config = copy.deepcopy(config)
        self.config.level = self.config.level_ori
        self.data_collector = build_data_collector(self.config)
        self.cell_processor = CellProcessor(self.data_collector.scope)
        self.primitive_hook_service = PrimitiveHookService(self)
        self.switch = False
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

    def check_level_valid(self):
        if self.config.level == Const.LEVEL_L2:
            raise MsprobeException(
                MsprobeException.INVALID_PARAM_ERROR, "L2 level dump function is currently not supported."
            )

    def build_hook(self, target_type, name):
        def forward_hook(api_or_cell_name, cell, input, output):
            if not self.should_excute_hook():
                return None

            if target_type == BaseScope.Module_Type_Module:
                api_or_cell_name = cell.mindstudio_reserved_name
                module_input_output = ModuleForwardInputsOutputs(args=input, kwargs={}, output=output)
            else:
                module_input_output = ModuleForwardInputsOutputs(args=input, kwargs=cell.input_kwargs,
                                                                 output=output)

            self.data_collector.update_api_or_module_name(api_or_cell_name)
            self.data_collector.forward_data_collect(api_or_cell_name, cell, pid, module_input_output)
            if self.data_collector.if_return_forward_new_output():
                return self.data_collector.get_forward_new_output()
            if target_type == BaseScope.Module_Type_API:
                del cell.input_kwargs
            return output

        def backward_hook(api_or_cell_name, cell, grad_input, grad_output):
            if not self.should_excute_hook():
                return

            if target_type == BaseScope.Module_Type_Module:
                api_or_cell_name = cell.mindstudio_reserved_name
            self.data_collector.update_api_or_module_name(api_or_cell_name)
            if self.data_collector:
                # 框架最新接口变更，grad_input和grad_output的含义发生了变化，与torch含义保持一致，因此此处调换顺序传入
                module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_output, grad_output=grad_input)
                self.data_collector.backward_data_collect(api_or_cell_name, cell, pid, module_input_output)

        pid = os.getpid()
        forward_name_template = name + Const.FORWARD
        backward_name_template = name + Const.BACKWARD
        forward_hook = functools.partial(forward_hook, forward_name_template)
        backward_hook = functools.partial(backward_hook, backward_name_template)

        def wrap_forward_hook(cell, input, output):
            return forward_hook(cell, input, output)

        def wrap_backward_hook(cell, grad_input, grad_output):
            return backward_hook(cell, grad_input, grad_output)

        return wrap_forward_hook, wrap_backward_hook


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
            NewPrimitive = type('NewPrimitive', (primitive.__class__,),
                                {'__call__': self.primitive_hook_service.wrap_primitive(primitive.__call__, pname)})
            primitive.__class__ = NewPrimitive

    def step(self):
        self.current_iter += 1
        self.data_collector.update_iter(self.current_iter)
        HOOKCell.cell_count = defaultdict(int)
        CellProcessor.reset_cell_stats()
        self.primitive_hook_service.primitive_counters.clear()
        self.data_collector.data_writer.reset_cache()
        JitDump.jit_count = defaultdict(int)

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

    def forward_backward_dump_end(self):
        if self.should_stop_service:
            return
        logger.info(f"{Const.TOOL_NAME}: debugger.forward_backward_dump_end() is set successfully. ")
        if not self.start_call:
            logger.error(f"{Const.TOOL_NAME}: debugger.start() is not set in the current scope.")
            raise Exception("debugger.start() is not set in the current scope.")
        if not self.switch:
            logger.error(f"{Const.TOOL_NAME}: debugger.forward_backward_dump_end() should be called between "
                         "debugger.start() and debugger.stop() ")
            raise Exception("debugger.stop() is already called. ")
        if self.config.step and self.current_iter not in self.config.step:
            return
        if self.config.rank and self.current_rank not in self.config.rank:
            return
        self.primitive_switch = False
        api_register.api_set_ori_func()

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

    def should_excute_hook(self):
        if not self.switch:
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
                forward_hook, backward_hook = self.build_hook(BaseScope.Module_Type_Module, prefix)
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
