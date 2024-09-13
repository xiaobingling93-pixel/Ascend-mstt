import os

from mindspore import Tensor
from mindspore.common.api import _MindsporeFunctionExecutor
from mindspore._c_expression import PyNativeExecutor_

from msprobe.mindspore.dump.hook_cell.api_registry import api_register
from msprobe.core.data_dump.data_processor.base import ModuleForwardInputsOutputs
from msprobe.core.common.const import Const


def dump_jit(name, in_feat, out_feat, is_forward):
    pid = os.getpid()
    ori_args = str(name)
    index = ori_args.find("<")
    if index != 0 and index != -1:
        result = ori_args[0:index]
    else:
        result = "JitFunction"
    if is_forward:
        name_template = "Jit." + result + ".forward"
    else:
        name_template = "Jit." + result + ".backward"
    if JitDump.need_dump():
        JitDump.data_collector.update_api_or_module_name(name_template)
        module_input_output = ModuleForwardInputsOutputs(args=in_feat, kwargs={}, output=out_feat)
        JitDump.data_collector.forward_data_collect(name_template, {}, pid, module_input_output)


class JitDump(_MindsporeFunctionExecutor):
    dump_config = None
    jit_enable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = PyNativeExecutor_.get_instance()

    def __call__(self, *args, **kwargs):
        api_register.api_set_ori_func()
        out = super().__call__(*args, **kwargs)
        if isinstance(args[0], Tensor):
            dump_jit({}, args, out, True)
        else:
            dump_jit(args[0], args[1:], out, True)
        JitDump.jit_enable = True
        api_register.api_set_hook_func()
        return out

    @classmethod
    def set_config(cls, value):
        cls.dump_config = value

    @classmethod
    def set_data_collector(cls, value):
        cls.data_collector = value

    @classmethod
    def need_dump(cls):
        if cls.dump_config.task != Const.TENSOR and cls.dump_config.task != Const.STATISTICS:
            return False
        if not cls.data_collector or cls.data_collector.data_processor.is_terminated:
            return False
        return True

    def grad(self, obj, grad, weights, grad_position, *args,  **kwargs):
        if JitDump.jit_enable:
            api_register.api_set_ori_func()
        output = self._executor.grad(grad, obj, weights, grad_position, *args, *(kwargs.values()))
        if JitDump.jit_enable:
            dump_jit(obj, args, None, False)
            api_register.api_set_hook_func()
        return output
