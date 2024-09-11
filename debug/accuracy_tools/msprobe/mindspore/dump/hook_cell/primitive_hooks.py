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


class PrimitiveHook:
    def __init__(self, service_instance, primitive_name):
        self.service_instance = service_instance
        self.primitive_name = primitive_name
        self.pid = os.getpid()

    def create_backward_hook(self, captured_grads, num_tensors, updated_primitive_name, hook_type):
        def backward_hook(grad):
            captured_grads.append(grad)
            backward_primitive_name = f"{updated_primitive_name}.{Const.BACKWARD}"
            if len(captured_grads) == num_tensors:
                try:
                    if hook_type == Const.INPUT:
                        self._collect_backward_input_data(backward_primitive_name, captured_grads)
                    elif hook_type == Const.OUTPUT:
                        self._collect_backward_output_data(backward_primitive_name, captured_grads)
                except Exception as exception:
                    raise Exception(f"Primitive backward hook error: {exception}, "
                                    f"primitive_name: {updated_primitive_name}") from exception
                captured_grads.clear()
        return backward_hook

    def _collect_backward_input_data(self, primitive_name, captured_grads):
        new_module_input_output = ModuleBackwardInputs(grad_input=tuple(captured_grads))
        self.service_instance.data_collector.backward_input_data_collect(
            primitive_name, self.service_instance, self.pid, new_module_input_output)

    def _collect_backward_output_data(self, primitive_name, captured_grads):
        new_module_input_output = ModuleBackwardOutputs(grad_output=tuple(captured_grads))
        self.service_instance.data_collector.backward_output_data_collect(
            primitive_name, self.service_instance, self.pid, new_module_input_output)

    def hook_inputs(self, args, captured_grads_input, updated_primitive_name):
        num_tensors = sum(isinstance(arg, Tensor) for arg in args)
        input_backward_hook = self.create_backward_hook(captured_grads_input, num_tensors, updated_primitive_name, Const.INPUT)
        hooked_inputs = [ops.HookBackward(input_backward_hook)(arg) if isinstance(arg, Tensor) else arg for arg in args]
        return hooked_inputs

    def hook_outputs(self, out, captured_grads_output, updated_primitive_name):
        if isinstance(out, tuple):
            num_output_tensors = sum(isinstance(tensor, Tensor) for tensor in out)
        else:
            num_output_tensors = 1
        output_backward_hook = self.create_backward_hook(captured_grads_output, num_output_tensors, updated_primitive_name, Const.OUTPUT)

        if isinstance(out, Tensor):
            return ops.HookBackward(output_backward_hook)(out)
        elif isinstance(out, tuple):
            return tuple(ops.HookBackward(output_backward_hook)(tensor) if isinstance(tensor, Tensor) else tensor for tensor in out)
        return out


class PrimitiveWrapper:
    def __init__(self, service_instance, primitive_name):
        self.service_instance = service_instance
        self.primitive_name = primitive_name
        self.hook_manager = PrimitiveHook(service_instance, primitive_name)

    def wrapped_primitive_call(self, origin_func, instance_self, *args, **kwargs):
        # 更新 primitive 调用计数
        self.service_instance.update_primitive_counters(self.primitive_name)
        current_count = self.service_instance.primitive_counters.get(self.primitive_name, 0)
        updated_primitive_name = f"{Const.PRIMITIVE_PREFIX}.{self.primitive_name}.{current_count}"

        # 检查 dump 开关是否打开
        if not self.service_instance.switch:
            return origin_func(*args, **kwargs)

        captured_grads_input, captured_grads_output = [], []

        try:
            # 钩住输入数据
            hooked_inputs = self.hook_manager.hook_inputs(args, captured_grads_input, updated_primitive_name)
            out = origin_func(*hooked_inputs, **kwargs)
        except Exception as exception:
            raise Exception(f"Error during primitive call: {exception}, primitive_name: {self.primitive_name}") from exception

        # 前向数据收集
        forward_primitive_name = f"{updated_primitive_name}.{Const.FORWARD}"
        self.service_instance.data_collector.visit_and_clear_overflow_status(forward_primitive_name)
        if self.service_instance.data_collector:
            module_input_output = ModuleForwardInputsOutputs(args=hooked_inputs, kwargs=kwargs, output=out)
            self.service_instance.data_collector.forward_data_collect(forward_primitive_name, instance_self, os.getpid(), module_input_output)

        # 钩住输出数据
        out = self.hook_manager.hook_outputs(out, captured_grads_output, updated_primitive_name)
        return out

