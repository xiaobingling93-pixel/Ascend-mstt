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

from msprobe.core.data_dump.scope import ModuleRangeScope, MixRangeScope
from msprobe.core.common.const import Const


class CellProcessor:
    cell_count = {}
    cell_stack = []
    api_parent_node = ""
    module_node = {}

    def __init__(self, scope):
        self.scope = scope if isinstance(scope, (ModuleRangeScope, MixRangeScope)) else None

    @staticmethod
    def set_cell_count(cell_name):
        if cell_name not in CellProcessor.cell_count:
            CellProcessor.cell_count[cell_name] = 0
        else:
            CellProcessor.cell_count[cell_name] += 1
        return CellProcessor.cell_count[cell_name]

    @classmethod
    def reset_cell_stats(cls):
        cls.cell_count = {}
        cls.cell_stack = []
        cls.api_parent_node = ""
        cls.module_node = {}

    def node_hook(self, name_prefix, start_or_stop, **kwargs):
        def begin_hook(cell, input_data):
            full_name = self.set_and_get_reserved_name(cell, name_prefix, is_called_by_pre_hook=True)
            if CellProcessor.cell_stack:
                CellProcessor.module_node[full_name] = CellProcessor.cell_stack[-1]
            else:
                CellProcessor.module_node[full_name] = None

            CellProcessor.cell_stack.append(full_name)
            CellProcessor.api_parent_node = full_name

            if self.scope:
                self.scope.begin_module(full_name)

        def end_hook(cell, input_data, output_data):
            if CellProcessor.cell_stack:
                CellProcessor.cell_stack.pop()
            if CellProcessor.cell_stack:
                CellProcessor.api_parent_node = CellProcessor.cell_stack[-1]
            else:
                CellProcessor.api_parent_node = None

            if self.scope:
                self.scope.end_module(cell.mindstudio_reserved_name)

        return begin_hook if Const.START == start_or_stop else end_hook

    def set_and_get_reserved_name(self, cell, cell_name, is_called_by_pre_hook=False):
        if not is_called_by_pre_hook and hasattr(cell, 'has_pre_hook_called') and cell.has_pre_hook_called:
            cell.has_pre_hook_called = False
        else:
            if is_called_by_pre_hook:
                cell.has_pre_hook_called = True
            index = self.set_cell_count(cell_name)
            cell.mindstudio_reserved_name = cell_name + Const.SEP + str(index)
        return cell.mindstudio_reserved_name
