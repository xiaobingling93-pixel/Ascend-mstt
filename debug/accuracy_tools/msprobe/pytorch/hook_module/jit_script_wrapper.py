# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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

import torch

from msprobe.pytorch.hook_module.api_register import get_api_register


def wrap_jit_script_func():
    def patched_script(*args, **kwargs):
        all_api_registered = api_register.all_api_registered
        if all_api_registered:
            api_register.restore_all_api()
        result = original_script(*args, **kwargs)
        if all_api_registered:
            api_register.register_all_api()
        return result

    original_script = torch.jit.script
    api_register = get_api_register()
    torch.jit.script = patched_script
