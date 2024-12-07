# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

import os
import unittest

from msprobe.core.common.file_utils import load_yaml
from msprobe.mindspore.common.const import FreeBenchmarkConst


class TestSupportWrapOps(unittest.TestCase):
    def test_support_wrap_ops(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cur_path))))
        yaml_path = os.path.join(base_path, "mindspore", "free_benchmark", "data", "support_wrap_ops.yaml")

        supported_ops_list = load_yaml(yaml_path)
        for k, v in FreeBenchmarkConst.API_PREFIX_DICT.items():
            ops = supported_ops_list.get(k)
            self.assertIsNotNone(ops)
