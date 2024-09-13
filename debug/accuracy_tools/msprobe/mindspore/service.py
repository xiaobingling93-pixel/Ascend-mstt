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
from msprobe.core.common.utils import Const
from msprobe.core.common.exceptions import DistributedNotInitializedError
from msprobe.mindspore.dump.hook_cell.api_registry import api_register
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
        self.switch = False
        self.current_iter = 0
        self.first_start = True
        self.current_rank = None
        self.primitive_counters = {}
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
        if self.config.level == "L2":
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

    def wrap_primitive(self, origin_func, primitive_name):
        service_instance = self

        def create_backward_hook(captured_grads, num_tensors, updated_primitive_name, hook_type):
            def backward_hook(grad):
                captured_grads.append(grad)
                backward_primitive_name = f"{updated_primitive_name}.{Const.BACKWARD}"
                try:
                    if len(captured_grads) == num_tensors and hook_type == Const.INPUT:
                        service_instance.data_collector.update_api_or_module_name(backward_primitive_name)
                        new_module_input_output = ModuleBackwardOutputs(grad_output=tuple(captured_grads))
                        service_instance.data_collector.backward_output_data_collect(
                            backward_primitive_name, service_instance, os.getpid(), new_module_input_output
                        )
                        captured_grads.clear()
                    elif len(captured_grads) == num_tensors and hook_type == Const.OUTPUT:
                        service_instance.data_collector.update_api_or_module_name(backward_primitive_name)
                        new_module_input_output = ModuleBackwardInputs(grad_input=tuple(captured_grads))
                        service_instance.data_collector.backward_input_data_collect(
                            backward_primitive_name, service_instance, os.getpid(), new_module_input_output
                        )
                        captured_grads.clear()

                except Exception as exception:
                    raise Exception(f"This is a primitive op {hook_type}_backward dump error: {exception},"
                                    f" updated_primitive_name: {updated_primitive_name}") from exception

            return backward_hook

        def hook_primitive_inputs(args, captured_grads_input, updated_primitive_name):
            hooked_inputs = []
            num_tensors = sum(isinstance(arg, Tensor) for arg in args)
            input_backward_hook = create_backward_hook(captured_grads_input, num_tensors, updated_primitive_name,
                                                       Const.INPUT)
            for _, arg in enumerate(args):
                if isinstance(arg, Tensor):
                    arg_hooked = ops.HookBackward(input_backward_hook)(arg)
                    hooked_inputs.append(arg_hooked)
                else:
                    hooked_inputs.append(arg)
            return hooked_inputs

        def hook_primitive_outputs(out, captured_grads_output, updated_primitive_name):
            if isinstance(out, tuple):
                num_output_tensors = sum(isinstance(tensor, Tensor) for tensor in out)
            else:
                num_output_tensors = 1
            output_backward_hook = create_backward_hook(captured_grads_output, num_output_tensors,
                                                        updated_primitive_name, Const.OUTPUT)

            if isinstance(out, Tensor):
                return ops.HookBackward(output_backward_hook)(out)
            elif isinstance(out, tuple):
                hooked_outputs = []
                for tensor in out:
                    if isinstance(tensor, Tensor):
                        hooked_outputs.append(ops.HookBackward(output_backward_hook)(tensor))
                    else:
                        hooked_outputs.append(tensor)
                return tuple(hooked_outputs)
            return out

        def wrapped_primitive_call(instance_self, *args, **kwargs):
            service_instance.update_primitive_counters(primitive_name)
            current_count = service_instance.primitive_counters.get(primitive_name, 0)
            updated_primitive_name = f"{Const.PRIMITIVE_PREFIX}.{primitive_name}.{current_count}"

            if not service_instance.switch:
                return origin_func(*args, **kwargs)

            captured_grads_input, captured_grads_output = [], []

            try:
                hooked_inputs = hook_primitive_inputs(args, captured_grads_input, updated_primitive_name)
            except Exception as exception:
                raise Exception("This is a primitive op dump error during input hooking: {},"
                                " primitive_name: {}".format(exception, primitive_name)) from exception

            try:
                out = origin_func(*hooked_inputs, **kwargs)
            except Exception as exception:
                raise Exception("This is a primitive op dump error during function call: {},"
                                " primitive_name: {}".format(exception, primitive_name)) from exception

            forward_primitive_name = f"{updated_primitive_name}.{Const.FORWARD}"
            service_instance.data_collector.update_api_or_module_name(forward_primitive_name)
            if service_instance.data_collector:
                module_input_output = ModuleForwardInputsOutputs(args=hooked_inputs, kwargs=kwargs, output=out)
                try:
                    service_instance.data_collector.forward_data_collect(forward_primitive_name, instance_self,
                                                                         os.getpid(), module_input_output)
                except Exception as exception:
                    raise Exception("This is a primitive op dump error during forward data collection: {},"
                                    " primitive_name: {}".format(exception, primitive_name)) from exception

                if service_instance.data_collector.if_return_forward_new_output():
                    out = service_instance.data_collector.get_forward_new_output()

            try:
                out = hook_primitive_outputs(out, captured_grads_output, updated_primitive_name)
            except Exception as exception:
                raise Exception("This is a primitive op dump error during output hooking: {},"
                                " primitive_name: {}".format(exception, primitive_name)) from exception

            return out

        return wrapped_primitive_call

    def update_primitive_counters(self, primitive_name):
        if primitive_name not in self.primitive_counters:
            self.primitive_counters[primitive_name] = 0
        else:
            self.primitive_counters[primitive_name] += 1

    def register_hooks(self):
        primitive_set = set()
        for _, cell in self.model.cells_and_names():
            for pname, primitive in cell._primitives.items():
                primitive_set.add((pname, primitive))

        for pname, primitive in primitive_set:
            NewPrimitive = type('NewPrimitive', (primitive.__class__,),
                                {'__call__': self.wrap_primitive(primitive.__call__, pname)})
            primitive.__class__ = NewPrimitive

    def step(self):
        self.current_iter += 1
        self.data_collector.update_iter(self.current_iter)
        HOOKCell.cell_count = defaultdict(int)
        CellProcessor.cell_count = {}
        self.primitive_counters.clear()

    def start(self, model=None):
        self.start_call = True
        if self.should_stop_service:
            return
        if self.need_end_service():
            api_register.api_set_ori_func()
            self.should_stop_service = True
            self.switch = False
            logger.info("************************************************")
            logger.info(f"*          {Const.TOOL_NAME} ends successfully.          *")
            logger.info("************************************************")
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
            if self.config.level == "L1":
                JitDump.set_config(self.config)
                JitDump.set_data_collector(self.data_collector)
                ms.common.api._MindsporeFunctionExecutor = JitDump
                ms.common.api._PyNativeExecutor.grad = JitDump.grad
                if pijit_label:
                    PIJitCaptureContext.__enter__ = self.empty
                    PIJitCaptureContext.__exit__ = self.empty
            self.first_start = False

        self.switch = True
        logger.info(f"Dump switch is turned on at step {self.current_iter}. ")
        self.create_dirs()
        logger.info(f"Dump data will be saved in {self.dump_iter_dir}.")

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
        self.start_call = False
        self.data_collector.write_json()

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
        if self.config.level == "L1":
            api_register.initialize_hook(functools.partial(self.build_hook, BaseScope.Module_Type_API))
            api_register.api_set_hook_func()
            if self.model:
                self.register_hooks()

        if self.config.level == "L0":
            if not self.model:
                raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                       "The current level is L0, the model cannot be None")
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
