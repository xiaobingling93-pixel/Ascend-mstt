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

from collections import OrderedDict

from mindspore import Tensor
from mindspore.common.hook_handle import HookHandle
from mindspore.ops.operations import _inner_ops as inner

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.data_dump.scope import ModuleRangeScope, MixRangeScope, BaseScope
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.common.utils import (
    is_mindtorch,
    get_cells_and_names_with_index,
    has_kwargs_in_forward_hook,
    is_graph_mode_cell_dump_allowed
)
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.graph_mode_cell_dump import GraphModeCellDump
from msprobe.core.common.runtime import Runtime


def get_cell_construct(construct):
    def _construct(self, *args, **kwargs):
        if hasattr(self, 'msprobe_hook'):
            setattr(self, 'msprobe_input_kwargs', kwargs)
        return construct(self, *args, **kwargs)
    return _construct


class CellProcessor:
    cell_count = {}
    cell_stack = []
    api_parent_node = None
    module_node = {}
    cell_bw_hook_kernels = {}
    cell_backward_pre_hook = []
    cell_backward_hook = []

    def __init__(self, scope):
        self.scope = scope if isinstance(scope, (ModuleRangeScope, MixRangeScope)) else None

    @staticmethod
    def set_and_get_calls_number(cell_name):
        if cell_name not in CellProcessor.cell_count:
            CellProcessor.cell_count[cell_name] = 0
        else:
            CellProcessor.cell_count[cell_name] += 1
        return CellProcessor.cell_count[cell_name]

    @classmethod
    def reset_cell_stats(cls):
        cls.cell_count = {}
        cls.cell_stack = []
        cls.api_parent_node = None
        cls.module_node = {}
        cls.cell_bw_hook_kernels = {}
        cls.cell_backward_pre_hook = []
        cls.cell_backward_hook = []

    def register_cell_hook(self, models, build_hook, config: DebuggerConfig):
        if not models:
            raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR,
                                   'The model cannot be None, when level is "L0" or "mix"')

        is_registered = False
        model_type = Const.MODULE if is_mindtorch() else Const.CELL
        cells_with_index_in_pynative_mode, cells_with_index_in_graph_mode = get_cells_and_names_with_index(models)
        construct_name = '_call_impl' if is_mindtorch() else '_run_construct'

        for index, cells_and_names in cells_with_index_in_pynative_mode.items():
            model = models if index == "-1" else models[int(index)]
            for name, cell in cells_and_names:
                if cell == model:
                    continue

                if not has_kwargs_in_forward_hook():
                    if not hasattr(cell.__class__, 'msprobe_construct'):
                        setattr(cell.__class__, 'msprobe_construct', True)
                        if hasattr(cell.__class__, construct_name):
                            setattr(cell.__class__, construct_name,
                                    get_cell_construct(getattr(cell.__class__, construct_name)))
                setattr(cell, 'msprobe_hook', True)

                cell_index = (index + Const.SEP) if index != "-1" else ""
                prefix = f'{model_type}{Const.SEP}{cell_index}{name}{Const.SEP}{cell.__class__.__name__}{Const.SEP}'

                forward_pre_hook = self.build_cell_hook(prefix, build_hook)
                cell.register_forward_pre_hook(forward_pre_hook)

                if not is_registered:
                    logger.info("The cell hook function is successfully mounted to the model.")
                is_registered = True

        if is_graph_mode_cell_dump_allowed(config):
            cells_and_names_in_graph_mode = []
            for index, cells_and_names in cells_with_index_in_graph_mode.items():
                model = models if index == "-1" else models[int(index)]
                for name, cell in cells_and_names:
                    if cell == model:
                        continue
                    cell_index = (index + Const.SEP) if index != "-1" else ""
                    cells_and_names_in_graph_mode.append((f'{cell_index}{name}', cell))

            if cells_and_names_in_graph_mode:
                Runtime.run_mode = MsConst.PYNATIVE_GRAPH_MODE
                GraphModeCellDump(config, cells_and_names_in_graph_mode, strict=False).handle()

    def build_cell_hook(self, cell_name, build_data_hook):
        def forward_pre_hook(cell, args):
            index = CellProcessor.set_and_get_calls_number(cell_name)
            full_forward_name = f'{cell_name}{Const.FORWARD}{Const.SEP}{index}'
            full_backward_name = f'{cell_name}{Const.BACKWARD}{Const.SEP}{index}'

            self.set_construct_info_in_pre_hook(full_forward_name)

            if not hasattr(cell, 'msprobe_forward_hook'):
                if is_mindtorch():
                    cell.register_forward_hook(forward_hook, prepend=True, with_kwargs=True)
                else:
                    forward_hook_dict = getattr(cell, '_forward_hook', OrderedDict())
                    if has_kwargs_in_forward_hook():
                        forward_hook_with_kwargs_dict = getattr(cell, '_forward_hook_with_kwargs', OrderedDict())
                        handle = HookHandle(forward_hook_dict, extra_dict=forward_hook_with_kwargs_dict)
                        forward_hook_with_kwargs_dict[handle.handle_id] = True
                    else:
                        handle = HookHandle(forward_hook_dict)
                    forward_hook_dict[handle.handle_id] = forward_hook
                    forward_hook_dict.move_to_end(handle.handle_id, last=False)

                setattr(cell, 'msprobe_forward_hook', True)

            def get_backward_hook(backward_data_hook, full_backward_name):
                def backward_hook_fn(cell, grad_input, grad_output):
                    new_output = backward_data_hook(cell, grad_input, grad_output)
                    self.set_construct_info_in_hook(full_backward_name)
                    cell.has_pre_hook_called = False
                    return new_output
                return backward_hook_fn

            enable_hooked = sum(
                [isinstance(ele, Tensor) and ele.dtype not in MsConst.NonDifferentiableType for ele in args]
            )
            if enable_hooked:
                backward_hook = OrderedDict()
                hook_set = build_data_hook(BaseScope.Module_Type_Module, full_forward_name)
                backward_hook[full_backward_name] = get_backward_hook(hook_set.backward_hook, full_backward_name)
                CellProcessor.cell_backward_hook.append(backward_hook)
                bw_hook = inner.CellBackwardHook(full_backward_name, cell,
                                                 self.cell_backward_hook[-1])
                bw_hook.register_backward_hook()
                CellProcessor.cell_bw_hook_kernels[full_forward_name] = bw_hook

                args = bw_hook(*args)

            return args

        def forward_hook(cell, args, kwargs_or_output, output_or_kwargs=None):
            index = CellProcessor.cell_count.get(cell_name, 0)
            full_forward_name = f'{cell_name}{Const.FORWARD}{Const.SEP}{index}'
            full_backward_name = f'{cell_name}{Const.BACKWARD}{Const.SEP}{index}'

            self.set_construct_info_in_hook(full_forward_name)
            
            hook_set = build_data_hook(BaseScope.Module_Type_Module, full_forward_name)
            hook_result = hook_set.forward_hook(cell, args, kwargs_or_output, output_or_kwargs)
            if hook_result is not None:
                outputs = hook_result
            else:
                outputs = output_or_kwargs if has_kwargs_in_forward_hook() else kwargs_or_output

            bw_hook = CellProcessor.cell_bw_hook_kernels.get(full_forward_name)
            if bw_hook:
                if not isinstance(outputs, (Tensor, tuple)):
                    logger.warning("For backward hooks to be called,"
                                   " cell output should be a Tensor or a tuple of Tensors"
                                   f" but received {type(outputs)}")
                if isinstance(outputs, tuple):
                    new_outputs = bw_hook(*outputs)
                else:
                    new_outputs = bw_hook(outputs)
                if isinstance(outputs, tuple) and len(outputs) == 1:
                    new_outputs = (new_outputs,)
                outputs = new_outputs

            def get_backward_pre_hook(full_backward_name, backward_data_hook):
                def backward_pre_hook_fn(cell, grad_output):
                    cell.has_pre_hook_called = True
                    self.set_construct_info_in_pre_hook(full_backward_name)
                    if backward_data_hook:
                        backward_data_hook(cell, (), grad_output)
                        self.set_construct_info_in_hook(full_backward_name)
                        cell.has_pre_hook_called = False
                return backward_pre_hook_fn

            backward_pre_hook = OrderedDict()
            backward_data_hook = None if bw_hook else hook_set.backward_hook
            backward_pre_hook[full_backward_name] = get_backward_pre_hook(full_backward_name, backward_data_hook)
            CellProcessor.cell_backward_pre_hook.append(backward_pre_hook)
            bw_pre_hook = inner.CellBackwardHook(full_backward_name, cell,
                                                 self.cell_backward_pre_hook[-1])
            bw_pre_hook.register_backward_pre_hook()

            if isinstance(outputs, tuple):
                result = bw_pre_hook(*outputs)
            else:
                result = bw_pre_hook(outputs)
            if isinstance(outputs, tuple):
                if len(outputs) == 1:
                    result = (result,)
                if len(result) != len(outputs):
                    raise TypeError(
                        f"The backward pre hook return value size is {len(result)} "
                        f"not equal to output size {len(outputs)}"
                    )
            return result

        return forward_pre_hook

    def set_construct_info_in_pre_hook(self, full_name):
        if self.cell_stack:
            CellProcessor.module_node[full_name] = self.cell_stack[-1]
        else:
            CellProcessor.module_node[full_name] = None
        CellProcessor.cell_stack.append(full_name)
        CellProcessor.api_parent_node = full_name
        if self.scope:
            self.scope.begin_module(full_name)

    def set_construct_info_in_hook(self, full_name):
        if self.cell_stack:
            CellProcessor.cell_stack.pop()
        CellProcessor.api_parent_node = CellProcessor.cell_stack[-1] if self.cell_stack else None
        if self.scope:
            self.scope.end_module(full_name)
