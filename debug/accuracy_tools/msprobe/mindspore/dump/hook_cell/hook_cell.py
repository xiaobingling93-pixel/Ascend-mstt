# Copyright 2024 Huawei Technologies Co., Ltd
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
# ============================================================================

from collections import defaultdict

from mindspore import nn

from msprobe.core.common.const import Const


class HOOKCell(nn.Cell):
    cell_count = defaultdict(int)
    g_stop_hook = False

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

            HOOKCell.cell_count[self.prefix] += 1
            self.prefix = self.prefix + str(HOOKCell.cell_count[self.prefix] - 1) + Const.SEP
            forward_hook, backward_hook = build_hook(self.prefix)
            self.register_forward_hook(forward_hook)
            self.register_backward_hook(backward_hook)

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
