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

from mindspore import nn

from msprobe.mindspore.common.utils import is_mindtorch, register_backward_hook_functions


def add_cell_count(name):
    HOOKCell.cell_count[name] += 1


def get_cell_count(name):
    return HOOKCell.cell_count[name]


def __init__(self, build_hook) -> None:
    super(HOOKCell, self).__init__()
    self.changed_status = False
    self.input_kwargs = {}
    self.prefix = ""
    if not HOOKCell.g_stop_hook:
        HOOKCell.g_stop_hook = True
        self.changed_status = True
        if hasattr(self, "prefix_api_name"):
            self.prefix = self.prefix_api_name

        self.forward_data_collected = False
        forward_pre_hook, forward_hook, backward_hook, backward_pre_hook = build_hook(self.prefix)
        self.register_forward_pre_hook(forward_pre_hook)
        self.register_forward_hook(forward_hook)
        register_backward_hook_functions["full"](self, backward_hook)
        register_backward_hook_functions["pre"](self, backward_pre_hook)


# 重载call，加全局标志。
def __call__(self, *args, **kwargs):
    try:
        self.input_kwargs = kwargs
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
