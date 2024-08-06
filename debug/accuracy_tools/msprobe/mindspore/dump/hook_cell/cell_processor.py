
from msprobe.core.data_dump.scope import ModuleRangeScope


class CellProcessor:
    module_count = {}

    def __init__(self, scope):
        if isinstance(scope, ModuleRangeScope):
            self.scope = scope
        else:
            self.scope = None
        
    @staticmethod
    def module_count_func(module_name):
        if module_name not in ModuleProcessor.module_count:
            CellProcessor.module_count[module_name] = 0
        else:
            CellProcessor.module_count[module_name] += 1
        return CellProcessor.module_count[module_name]
    
    def node_hook(self, name_prefix, start_or_stop, **kwargs):
        def pre_hook(module, input, output=None):
            try:
                index = self.module_count_func(name_prefix)
            except IndexError as e:
                index = None
                pass
            module.mindstudio_reserved_name = full_name = name_prefix + Const.SEP + str(index)
            if self.scope:
                self.scope.begin_module(full_name)
    
    def end_hook(module, input, output=None):
            if self.scope:
                self.scope.end_module(module.mindstudio_reserved_name)

    return pre_hook if Const.START in start_or_stop else end_hook
