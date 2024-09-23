from msprobe.core.data_dump.scope import ModuleRangeScope
from msprobe.core.common.const import Const


class CellProcessor:
    cell_count = {}
    cell_stack = []
    api_parent_node = ""
    module_node = {}

    def __init__(self, scope):
        if isinstance(scope, ModuleRangeScope):
            self.scope = scope
        else:
            self.scope = None

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
        def begin_hook(cell, input):
            index = self.set_cell_count(name_prefix)
            cell.mindstudio_reserved_name = full_name = name_prefix + Const.SEP + str(index)
            if CellProcessor.cell_stack:
                CellProcessor.module_node[full_name] = CellProcessor.cell_stack[-1]
            else:
                CellProcessor.module_node[full_name] = None
            
            CellProcessor.cell_stack.append(full_name)
            CellProcessor.api_parent_node = full_name

            if self.scope:
                self.scope.begin_module(full_name)

        def end_hook(cell, input, output):
            if CellProcessor.cell_stack:
                CellProcessor.cell_stack.pop()
            if CellProcessor.cell_stack:
                CellProcessor.api_parent_node = CellProcessor.cell_stack[-1]
            else:
                CellProcessor.api_parent_node = None

            if self.scope:
                self.scope.end_module(cell.mindstudio_reserved_name)

        return begin_hook if Const.START == start_or_stop else end_hook
