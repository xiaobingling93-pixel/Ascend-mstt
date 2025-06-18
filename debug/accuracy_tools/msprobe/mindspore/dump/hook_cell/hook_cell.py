# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
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

from collections import defaultdict

import mindspore as ms
from mindspore import nn

from msprobe.core.common.runtime import Runtime
from msprobe.mindspore.common.utils import is_mindtorch, register_backward_hook_functions


ms_version = ms.__version__


def add_cell_count(name):
    HOOKCell.cell_count[name] += 1


def get_cell_count(name):
    return HOOKCell.cell_count[name]


def __init__(self, hook_build_func) -> None:
    super(HOOKCell, self).__init__()
    self.changed_status = False
    self.msprobe_input_kwargs = {}
    if not HOOKCell.g_stop_hook:
        HOOKCell.g_stop_hook = True
        self.changed_status = True
        self.forward_data_collected = False

        if not Runtime.is_running:
            return
        prefix = self.prefix_api_name if hasattr(self, "prefix_api_name") else ""
        if callable(hook_build_func):
            hook_set = hook_build_func(prefix)
            if ms_version < "2.6.0" and not is_mindtorch():
                getattr(self, "_forward_pre_hook", {})[id(self)] = hook_set.forward_pre_hook
                getattr(self, "_forward_hook", {})[id(self)] = hook_set.forward_hook
            else:
                self.register_forward_pre_hook(hook_set.forward_pre_hook)
                self.register_forward_hook(hook_set.forward_hook)
            register_backward_hook_functions["full"](self, hook_set.backward_hook)
            register_backward_hook_functions["pre"](self, hook_set.backward_pre_hook)


# 重载call，加全局标志。
def __call__(self, *args, **kwargs):
    try:
        self.msprobe_input_kwargs = kwargs
        out = super(HOOKCell, self).__call__(*args, **kwargs)
    except Exception as e:
        raise e
    finally:
        if self.changed_status:
            self.changed_status = False
            HOOKCell.g_stop_hook = False
    return out


hook_cell_dict = {
    "cell_count": defaultdict(int),
    "g_stop_hook": False,
    "add_cell_count": staticmethod(add_cell_count),
    "get_cell_count": staticmethod(get_cell_count),
    "__init__": __init__,
    "__call__": __call__
}

if is_mindtorch():
    import torch
    HOOKCell = type("HOOKCell", (torch.nn.Module,), hook_cell_dict)
else:
    HOOKCell = type("HOOKCell", (nn.Cell,), hook_cell_dict)
