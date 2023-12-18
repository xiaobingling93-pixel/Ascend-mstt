#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2023-2023. Huawei Technologies Co., Ltd. All rights reserved.
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
"""

import os
import yaml

from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileOpen

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with FileOpen(yaml_path, 'r') as f:
    Ops = yaml.safe_load(f)
    WrapFunctionalOps = Ops.get('functional')
    WrapTensorOps = Ops.get('tensor')
    WrapTorchOps = Ops.get('torch')