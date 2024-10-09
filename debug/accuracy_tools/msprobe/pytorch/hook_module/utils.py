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
from msprobe.core.common.file_utils import load_yaml


def get_ops():
    cur_path = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
    ops = load_yaml(yaml_path)
    wrap_functional = ops.get('functional')
    wrap_tensor = ops.get('tensor')
    wrap_torch = ops.get('torch')
    wrap_npu_ops = ops.get('torch_npu')
    return set(wrap_functional) | set(wrap_tensor) | set(wrap_torch) | set(wrap_npu_ops)
