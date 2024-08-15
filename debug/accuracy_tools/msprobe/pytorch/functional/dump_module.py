import torch.nn as nn
from msprobe.pytorch.common.log import logger
from msprobe.core.common.const import Const
from msprobe.pytorch.hook_module.api_registry import api_register
from msprobe.pytorch.debugger.precision_debugger import PrecisionDebugger
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.data_dump.scope import BaseScope

module_count = {}


def module_dump(module, dump_name):
    if not isinstance(module, nn.Module):
        logger.error("The parameter:module in module_dump is not a Module subclass.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    if not isinstance(dump_name, str):
        logger.error("The parameter:dump_name in module_dump is not a str type.")
        raise MsprobeException(MsprobeException.INVALID_PARAM_ERROR)
    api_register.api_originality()
    if dump_name not in module_count:
        module_count[dump_name] = 0
    else:
        module_count[dump_name] += 1
    dump_name = dump_name + Const.SEP + str(module_count.get(dump_name)) + Const.SEP

    pdg = PrecisionDebugger()
    _, forward_hook, backward_hook, _ = pdg.service.build_hook(BaseScope.Module_Type_Module, dump_name)
    module.register_forward_hook(forward_hook, with_kwargs=True)
    module.register_full_backward_hook(backward_hook)

    module.register_forward_pre_hook(pdg.service.module_processor.node_hook(dump_name + Const.FORWARD, Const.START))
    module.register_forward_hook(pdg.service.module_processor.node_hook(dump_name + Const.FORWARD, Const.STOP))
    module.register_full_backward_pre_hook(
        pdg.service.module_processor.node_hook(dump_name + Const.BACKWARD, Const.START))
    module.register_full_backward_hook(pdg.service.module_processor.node_hook(dump_name + Const.BACKWARD, Const.STOP))


def module_dump_end():
    api_register.api_modularity()
