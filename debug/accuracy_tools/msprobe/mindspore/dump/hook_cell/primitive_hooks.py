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

class PrimitiveHookService:
    def __init__(self, service_instance):
        self.service_instance = service_instance
        self.primitive_counters = {}

    def wrap_primitive(self, origin_func, primitive_name):
        def create_backward_hook(captured_grads, num_tensors, updated_primitive_name, hook_type):
            def backward_hook(grad):
                captured_grads.append(grad)
                backward_primitive_name = f"{updated_primitive_name}.{Const.BACKWARD}"
                try:
                    if len(captured_grads) == num_tensors and hook_type == Const.INPUT:
                        self.service_instance.data_collector.visit_and_clear_overflow_status(backward_primitive_name)
                        new_module_input_output = ModuleBackwardOutputs(grad_output=tuple(captured_grads))
                        self.service_instance.data_collector.backward_output_data_collect(
                            backward_primitive_name, self, os.getpid(), new_module_input_output
                        )
                        captured_grads.clear()
                    elif len(captured_grads) == num_tensors and hook_type == Const.OUTPUT:
                        self.service_instance.data_collector.visit_and_clear_overflow_status(backward_primitive_name)
                        new_module_input_output = ModuleBackwardInputs(grad_input=tuple(captured_grads))
                        self.service_instance.data_collector.backward_input_data_collect(
                            backward_primitive_name, self, os.getpid(), new_module_input_output
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
            self.update_primitive_counters(primitive_name)
            current_count = self.primitive_counters.get(primitive_name, 0)
            updated_primitive_name = f"{Const.PRIMITIVE_PREFIX}.{primitive_name}.{current_count}"

            if not self.service_instance.switch:
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
            self.service_instance.data_collector.visit_and_clear_overflow_status(forward_primitive_name)
            if self.service_instance.data_collector:
                module_input_output = ModuleForwardInputsOutputs(args=hooked_inputs, kwargs=kwargs, output=out)
                try:
                    self.service_instance.data_collector.forward_data_collect(forward_primitive_name, instance_self,
                                                             os.getpid(), module_input_output)
                except Exception as exception:
                    raise Exception("This is a primitive op dump error during forward data collection: {},"
                                    " primitive_name: {}".format(exception, primitive_name)) from exception

                if self.service_instance.data_collector.if_return_forward_new_output():
                    out = self.service_instance.data_collector.get_forward_new_output()

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


