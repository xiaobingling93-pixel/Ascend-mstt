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

import os
from collections import defaultdict

from mindspore import Tensor
from mindspore._c_expression import PyNativeExecutor_
from mindspore.common.api import _MindsporeFunctionExecutor

from msprobe.mindspore.dump.hook_cell.api_registry import api_register
from msprobe.core.data_dump.data_processor.base import ModuleForwardInputsOutputs, ModuleBackwardInputsOutputs
from msprobe.core.common.const import Const
from msprobe.core.data_dump.data_processor.base import ModuleForwardInputsOutputs
from msprobe.mindspore.dump.hook_cell.api_registry import api_register


def dump_jit(name, in_feat, out_feat, is_forward):
    pid = os.getpid()
    ori_args = str(name)
    index = ori_args.find("<")
    if index != 0 and index != -1:
        result = ori_args[0:index]
    else:
        result = "JitFunction"
    if JitDump.need_dump():
        if is_forward:
            JitDump.jit_count[result] += 1
            name_template = Const.JIT + Const.SEP + result + Const.SEP + str(JitDump.jit_count[result]) + Const.SEP + \
                            Const.FORWARD
            JitDump.data_collector.update_api_or_module_name(name_template)
            module_input_output = ModuleForwardInputsOutputs(args=in_feat, kwargs={}, output=out_feat)
            JitDump.data_collector.forward_data_collect(name_template, None, pid, module_input_output)
        else:
            name_template = Const.JIT + Const.SEP + result + Const.SEP + str(JitDump.jit_count[result]) + Const.SEP + \
                            Const.BACKWARD
            JitDump.data_collector.update_api_or_module_name(name_template)
            module_input_output = ModuleBackwardInputsOutputs(grad_input=in_feat ,grad_output=out_feat)
            JitDump.data_collector.backward_data_collect(name_template, None, pid, module_input_output)


class JitDump(_MindsporeFunctionExecutor):
    dump_config = None
    jit_enable = False
    jit_dump_switch = True
    jit_count = defaultdict(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = PyNativeExecutor_.get_instance()

    def __call__(self, *args, **kwargs):
        api_register.api_set_ori_func()
        out = super().__call__(*args, **kwargs)
        if JitDump.jit_dump_switch and len(args) > 0:
            dump_jit(args[0], args, out, True)
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

    def grad(self, obj, grad, weights, grad_position, *args, **kwargs):
        if JitDump.jit_dump_switch and JitDump.jit_enable:
            api_register.api_set_ori_func()
        output = self._executor.grad(grad, obj, weights, grad_position, *args, *(kwargs.values()))
        if JitDump.jit_dump_switch and JitDump.jit_enable:
            dump_jit(obj, args, None, False)
            api_register.api_set_hook_func()
        return output
