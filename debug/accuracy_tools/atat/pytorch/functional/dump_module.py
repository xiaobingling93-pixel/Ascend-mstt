import torch.nn as nn
from atat.core.utils import print_error_log, DumpException
from .scope import BaseScope
from ..common.utils import Const
from ..hook_module.api_registry import api_register
from ..debugger.precision_debugger import PrecisionDebugger

module_count = {}


def module_dump(module, dump_name):
    if not isinstance(module, nn.Module):
        print_error_log("The parameter:module in module_dump is not a Module subclass.")
        raise DumpException(DumpException.INVALID_PARAM_ERROR)
    if not isinstance(dump_name, str):
        print_error_log("The parameter:dump_name in module_dump is not a str type.")
        raise DumpException(DumpException.INVALID_PARAM_ERROR)
    api_register.api_originality()
    if dump_name not in module_count:
        module_count[dump_name] = 0
    else:
        module_count[dump_name] += 1
    dump_name = dump_name + Const.SEP + str(module_count.get(dump_name)) + Const.SEP

    pdg = PrecisionDebugger()
    _, forward_hook, backward_hook = pdg.service.build_hook(BaseScope.Module_Type_Module, dump_name)
    module.register_forward_hook(forward_hook, with_kwargs=True)
    module.register_full_backward_hook(backward_hook)

    module.register_forward_pre_hook(pdg.service.module_processor.node_hook(dump_name + Const.FORWARD, Const.START))
    module.register_forward_hook(pdg.service.module_processor.node_hook(dump_name + Const.FORWARD, Const.STOP))
    module.register_full_backward_pre_hook(
        pdg.service.module_processor.node_hook(dump_name + Const.BACKWARD, Const.START))
    module.register_full_backward_hook(pdg.service.module_processor.node_hook(dump_name + Const.BACKWARD, Const.STOP))


def module_dump_end():
    api_register.api_modularity()
